# Run the following commands to start the GLM cluster.

cd ~/vllm-cluster && source glm/cluster-env.sh && make head PROFILE=glm     # Node 1
cd ~/vllm-cluster && source glm/cluster-env.sh && make worker PROFILE=glm   # Node 2
cd ~/vllm-cluster && make serve PROFILE=glm                                 # Node 1, new terminal

