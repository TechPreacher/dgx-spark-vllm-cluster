echo 'Checking Docker'
echo
echo 'Next command, expected output: cgroupfs'
docker info | grep -i "Cgroup Driver" # want: cgroupfs
echo
echo 'Next command, expected output: GPU 0: NVIDIA GB10 ...'
docker exec $(docker ps --filter name=^node- -q) nvidia-smi -L # want: GPU 0: NVIDIA GB10 ...
