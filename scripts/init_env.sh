echo -e "set number\nset mouse=a\nset encoding=utf-8" > $HOME/.vimrc
apt-get update && apt-get install -y tmux nvtop
echo -e  "set -g mouse on\n"> $HOME/.tmux.conf