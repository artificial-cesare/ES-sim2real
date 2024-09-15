#miniconda + mpi WSL logs

If you'd prefer that conda's base environment not be activated on startup,
   run the following command when conda is activated:

conda config --set auto_activate_base false

You can undo this by running `conda init --reverse $SHELL`

To activate conda's base environment in your current shell session:

eval "$(/home/artificial_cesare/miniconda3/bin/conda shell.YOUR_SHELL_NAME hook)"

To install conda's shell functions for easier access, first activate, then:

conda init

to show hidden files: 

ls -a

nano the .bashrc to 
export PATH="$HOME/miniconda3/bin:$PATH"

guide: 
https://enccs.github.io/intermediate-mpi/setup/ y
