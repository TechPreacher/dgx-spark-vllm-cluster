# on each node:
echo '{ "exec-opts": ["native.cgroupdriver=cgroupfs"] }' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker
