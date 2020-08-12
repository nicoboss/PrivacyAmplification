#!/bin/bash
tmux kill-session -t PrivacyAmplification
if [ "$1" = 'stop' ]; then
    echo  "Stopped!"
    exit 0
fi
if ! make -j 4; then
    exit 1
fi
cd build
tmux new-session -d -s PrivacyAmplification
tmux send-keys -t PrivacyAmplification 'tmux split-window -h' Enter
tmux send-keys -t PrivacyAmplification 'tmux split-window -v' Enter
tmux send-keys -t PrivacyAmplification 'tmux select-pane -t 0' Enter
tmux send-keys -t PrivacyAmplification 'tmux split-window -v' Enter
tmux send-keys -t PrivacyAmplification 'tmux select-pane -t 0' Enter
sleep 0.1
tmux send-keys -t PrivacyAmplification 'tmux send-keys -t 1 C-z ./SendKeysExample Enter' Enter && sleep 0.1
tmux send-keys -t PrivacyAmplification 'tmux send-keys -t 2 C-z ./MatrixSeedServerExample Enter' Enter && sleep 0.1
tmux send-keys -t PrivacyAmplification 'tmux send-keys -t 3 C-z ./ReceiveAmpOutExample Enter' Enter && sleep 0.1
tmux send-keys -t PrivacyAmplification 'clear' Enter && sleep 0.1
tmux send-keys -t PrivacyAmplification './PrivacyAmplification' Enter && sleep 0.1
tmux attach -t PrivacyAmplification
