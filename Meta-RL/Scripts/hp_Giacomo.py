## codice training usato per quel run (quindi con nome env e random seed inclusi), 
## final model checkpoint, env stats - vecnorm save

policy_kwargs = dict(log_std_init=-2,
                            ortho_init=True,
                            activation_fn=nn.Tanh,
                            net_arch=dict(pi=[256, 128], vf=[256, 128])
                            )

env = make_vec_env(args.env, n_envs=16, seed=args.seed, vec_env_cls=SubprocVecEnv)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

if args.env in ['Ant-v4', 'Humanoid-v4']:
            model = PPO("MlpPolicy",
                        env,
                        verbose=1,
                        policy_kwargs=policy_kwargs,
                        n_steps=256,
                        batch_size=256,
                        n_epochs=3,
                        gamma=0.995,
                        ent_coef=0.0,
                        tensorboard_log="tensorboard/",
                        sde_sample_freq=-1,
                        use_sde=False) ## but, no sde for dom rand?
else: # hopper-v4, halfcheetah-v4
    model = PPO("MlpPolicy",
                env,
                verbose=1,
                policy_kwargs=policy_kwargs,
                n_steps=256,
                batch_size=256,
                n_epochs=3,
                gamma=0.995,
                ent_coef=0.0,
                tensorboard_log="tensorboard/",
                sde_sample_freq=4,
                use_sde=True) ## but, no sde for dom rand?