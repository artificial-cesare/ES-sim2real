class DomainRandDistribution():
    """Handles Domain Randomization distributions"""

    def __init__(self,
                 dr_type: str,
                 distr: List[Dict]):
        self.set(dr_type, distr)
        return

    def set(self, dr_type, distr):
        if dr_type == 'beta':
            """
                distr: list of dict
                       4 keys per dimensions are expected:
                        m=min, M=max, a, b

                    Y ~ Beta(a,b,m,M)
                    y = x(M-m) + m
                    f(y) = f_x((y-m)/(M-m))/(M-m)
            """
            self.distr = distr.copy()
            self.ndims = len(self.distr)
            self.to_distr = []
            self.parameters = torch.zeros((self.ndims*2), dtype=torch.float32)
            for i in range(self.ndims):
                self.parameters[i*2] = float(distr[i]['a'])
                self.parameters[i*2 + 1] = float(distr[i]['b'])
                self.to_distr.append(Beta(self.parameters[i*2], self.parameters[i*2 + 1]))
        else:
            raise Exception('Unknown dr_type:'+str(dr_type))
        
        self.dr_type = dr_type
        return

    def sample(self, n_samples=1):
        if self.dr_type == 'beta':
            values = []
            for i in range(self.ndims):
                m, M = self.distr[i]['m'], self.distr[i]['M']
                values.append(self.to_distr[i].sample(sample_shape=(n_samples,)).numpy()*(M - m) + m)
            return np.array(values).T

    def sample_univariate(self, i, n_samples=1):
        if self.dr_type == 'beta':
            values = []
            m, M = self.distr[i]['m'], self.distr[i]['M']
            values.append(self.to_distr[i].sample(sample_shape=(n_samples,)).numpy())  # *(M - m) + m
            return np.array(values).T

    def _univariate_pdf(self, x, i, log=False, to_distr=None, standardize=False):
        """
            Computes univariate pdf(value) for
            i-th independent variable

            to_distr: custom torch univariate distribution list
            standardize: compute beta pdf in standard interval [0, 1]
        """
        to_distr = self.to_distr if to_distr is None else to_distr

        if self.dr_type == 'beta':
            m, M = self.distr[i]['m'], self.distr[i]['M']
            if np.isclose(M-m, 0):
                return np.isclose(x, m).astype(int)  # 1 if x = m = M, 0 otherwise
            else:
                if log:
                    if standardize:
                        return to_distr[i].log_prob(torch.tensor(x))
                    else:
                        return to_distr[i].log_prob(torch.tensor((x-m)/(M-m))) - torch.log(torch.tensor(M-m))
                else:
                    if standardize:
                        return torch.exp(to_distr[i].log_prob(torch.tensor(x)))
                    else:
                        return torch.exp(to_distr[i].log_prob(torch.tensor((x-m)/(M-m))))/(M-m)

        return

    def pdf(self, x, log=False, requires_grad=False, standardize=False, to_params=None):
        """
            Computes pdf(x)

            x: torch.tensor (Batch x ndims)
            log: compute the log(pdf(x))
            requires_grad: keep track of gradients w.r.t. beta params
            standardize: compute pdf in the standard [0, 1] interval,
                         by rescaling the input value
        """
        assert len(x.shape) == 2, 'Input tensor is expected with dims (batch, ndims)'
        density = torch.zeros(x.shape[0]) if log else torch.ones(x.shape[0])
        custom_to_distr = None
        if requires_grad:
            custom_to_distr, to_params = self._to_distr_with_grad(self, to_params=to_params)

        if standardize:
            x = self._standardize_value(x)

        for i in range(self.ndims):
            if log:
                density += self._univariate_pdf(x[:, i], i, log=True, to_distr=custom_to_distr, standardize=standardize)
            else:
                density *= self._univariate_pdf(x[:, i], i, log=False, to_distr=custom_to_distr, standardize=standardize)

        if requires_grad:
            return density, to_params
        else:
            return density

    def _standardize_value(self, x):
        """Linearly scale values from [m, M] to [0, 1]"""
        norm_x = x.copy()
        for i in range(self.ndims):
            m, M = self.distr[i]['m'], self.distr[i]['M']
            norm_x[:, i] =  (x[:, i] - m) / (M - m)
        return norm_x

    def kl_divergence(self, q, requires_grad=False, p_params=None, q_params=None):
        """Returns KL_div(self || q)

            q: DomainRandDistribution
            requires_grad: compute computational graph w.r.t.
                           beta parameters
        """
        assert isinstance(q, DomainRandDistribution)
        assert self.dr_type == q.dr_type 
        
        if self.dr_type == 'beta':
            if requires_grad:
                p_distr, p_params = self._to_distr_with_grad(self, to_params=p_params)
                q_distr, q_params = self._to_distr_with_grad(q,    to_params=q_params)
            else:
                p_distr = self.to_distr
                q_distr = q.to_distr

            kl_div = 0
            for i in range(self.ndims):
                # KL does not depend on loc params [m, M]
                kl_div += torch.distributions.kl_divergence(p_distr[i], q_distr[i])

            if requires_grad:
                return kl_div, p_params, q_params
            else:
                return kl_div

    def entropy(self, standardize=False):
        """Returns entropy of distribution"""
        if self.dr_type == 'beta':
            entropy = 0
            for i in range(self.ndims):
                if standardize:
                    entropy += self.to_distr[i].entropy()
                else:
                    # Y = aX + b => H(Y) = H(X) + log(a) 
                    m, M = self.distr[i]['m'], self.distr[i]['M']
                    entropy += self.to_distr[i].entropy() + torch.log(torch.tensor(M-m))

            return entropy

    def _to_distr_with_grad(self, p, to_params=None):
        """
            Returns list of torch Beta distributions
            given a DomainRandDistribution object p
        """
        if to_params is None:
            params = p.get_stacked_params()
            to_params = torch.tensor(params, requires_grad=True)

        to_distr = []
        for i in range(self.ndims):
            to_distr.append(Beta(to_params[i*2], to_params[i*2 + 1]))
        return to_distr, to_params

    def update_parameters(self, params):
        """Update the current beta parameters"""
        if self.dr_type == 'beta':
            distr = deepcopy(self.distr)
            for i in range(self.ndims):
                distr[i]['a'] = params[i*2]
                distr[i]['b'] = params[i*2 + 1]

            self.set(dr_type=self.get_dr_type(), distr=distr)

    def get(self):
        return self.distr

    def get_stacked_bounds(self):
        return np.array([[item['m'], item['M']] for item in self.distr]).reshape(-1)

    def get_stacked_params(self):
        return self.parameters.detach().numpy()

    def get_params(self):
        return self.parameters

    def get_dr_type(self):
        return self.dr_type

    def visualize_distr(self, ax=None, only_dims=None, **plot_kwargs):
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=self.ndims, figsize=(8,5))
            assert only_dims is None

        axes = [ax] if not isinstance(ax, np.ndarray) else ax  # handle case of ax not being list if it's single figure/dim
        only_dims = only_dims if only_dims is not None else list(range(self.ndims))  # include all dimensions
        for j, i in enumerate(only_dims):
            x = np.linspace(self.distr[i]['m'], self.distr[i]['M'], 100)
            axes[j].plot(x, self._univariate_pdf(x, i), **{'lw': 3, 'alpha':0.6, 'label': f'beta pdf dim{i}', **plot_kwargs})

    def print(self):
        if self.dr_type == 'beta':
            for i in range(self.ndims):
                print(f'dim{i}:', self.distr[i])

    def to_string(self):
        string = ''
        if self.dr_type == 'beta':
            for i in range(self.ndims):
                string += f"dim{i}: {self.distr[i]} | "
        return string

    @staticmethod
    def beta_from_stacked(stacked_bounds: np.ndarray, stacked_params: np.ndarray):
        """Creates instance of this class from the given stacked
            array of parameters

            stacked_bounds: beta boundaries [m_1, M_1, m_2, M_2, ...]
            stacked_params: beta parameters [a_1, b_1, a_2, b_2, ...]
        """
        distr = []
        ndim = stacked_bounds.shape[0]//2
        for i in range(ndim):
            d = {}
            d['m'] = stacked_bounds[i*2]
            d['M'] = stacked_bounds[i*2 + 1]
            d['a'] = stacked_params[i*2]
            d['b'] = stacked_params[i*2 + 1]
            distr.append(d)
        return DomainRandDistribution(dr_type='beta', distr=distr)

    @staticmethod
    def sigmoid(x, lb=0, up=1):
        """sigmoid of x"""
        x = x if torch.is_tensor(x) else torch.tensor(x)
        sig = (up-lb)/(1+torch.exp(-x)) + lb
        return sig

    @staticmethod
    def inv_sigmoid(x, lb=0, up=1):
        """return sigmoid^-1(x)"""
        x = x if torch.is_tensor(x) else torch.tensor(x)
        assert torch.all(x <= up) and torch.all(x >= lb)
        inv_sig = -torch.log((up-lb)/(x-lb) - 1)
        return inv_sig