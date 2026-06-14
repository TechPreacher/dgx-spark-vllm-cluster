# On Node 1, start head node

# All four ConnectX-7 ports are used as a data plane (~800 GbE aggregate).
# PRIMARY_IF carries the Ray control plane (single IP for VLLM_HOST_IP/MASTER_ADDR).
export PRIMARY_IF=enp1s0f1np1
export DATA_IFS=enp1s0f0np0,enp1s0f1np1,enP2p1s0f0np0,enP2p1s0f1np1

export VLLM_HOST_IP=$(ip -4 addr show $PRIMARY_IF | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
export VLLM_IMAGE=nvcr.io/nvidia/vllm:25.11-py3

echo "Primary (control) interface: $PRIMARY_IF  IP: $VLLM_HOST_IP"
echo "Data-plane interfaces:       $DATA_IFS"

bash run_cluster.sh $VLLM_IMAGE $VLLM_HOST_IP --head ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$DATA_IFS \
  -e NCCL_SOCKET_IFNAME=$DATA_IFS \
  -e NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1 \
  -e NCCL_CROSS_NIC=1 \
  -e OMPI_MCA_btl_tcp_if_include=$DATA_IFS \
  -e GLOO_SOCKET_IFNAME=$DATA_IFS \
  -e TP_SOCKET_IFNAME=$PRIMARY_IF \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$VLLM_HOST_IP \
  -e NCCL_DEBUG=INFO \
  -e NCCL_DEBUG_SUBSYS=INIT,NET
