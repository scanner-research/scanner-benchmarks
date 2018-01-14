if [ -z "$1" ]
  then
    echo "Usage: cinematography_spawn_nodes_gpu.sh <num_nodes>"
    exit
fi

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

num_nodes=$1
num_gpus=2
num_simul=100

SSH_KEY="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC+jBMU/O4ePOmxVoodKC/dz5+G4uyz2PmPdOSBFDkZJjHSvIlj+yKHCfM8uFQLct0mSg3jHy7RZDCSIcuXX99wBoWrFwNns8ODwAULYZt6aw8Yk6tHnG4gtz6nKM0k/AxqRFXqxZ94ON8YzOxY/hXNW1syFsIrbryBK2py8xHcn86pdwuSkGvEW5TyMG9AnngnITfOzgVENJ7RE0kLaxD295jMkJiM5XfVvhrcfs1X+YHHZTA2fYLI73LmCtkBFe2M8R9BTKsyv7myhnMjv4V/RxI7I6Z/XGXWvVTfb7GyTs7Dy+RDJa3z9SG+q7jeDUPcssZ3osJjUce+To1MYqi1 apoms@Alexanders-MacBook-Pro.local"

max=$(($num_nodes + $num_simul - 1))
for j in `seq 0 $num_simul $max`; do
    start=$(($j + 1))
    end=$(($j + $num_simul))
    m=$(( $num_nodes < $end ? $num_nodes : $end ))
    echo "Creating disks $start through $m..."
    for i in `seq $start 1 $m`; do
        gcloud compute --project "visualdb-1046" \
               disks create "scanner-apoms-$i" \
               --size "30" \
               --zone "us-east1-d" \
               --source-snapshot "scanner-apoms-siggraph2018-smaller-gpu" \
               --type "pd-ssd" &
    done
    wait
done


for j in `seq 0 $num_simul $max`; do
    start=$(($j + 1))
    end=$(($j + $num_simul))
    m=$(( $num_nodes < $end ? $num_nodes : $end ))
    echo "Creating instances $start through $m..."
    for i in `seq $start 1 $m`; do
        gcloud compute --project "visualdb-1046" \
               instances create "scanner-apoms-$i" \
               --zone "us-east1-d" \
               --machine-type "n1-highmem-16" \
               --network "default" \
               --metadata "ssh-keys=apoms:$SSH_KEY,node_id=$i" \
               --maintenance-policy "TERMINATE" \
               --service-account "50518136478-compute@developer.gserviceaccount.com" \
               --scopes "https://www.googleapis.com/auth/cloud-platform" \
               --tags "http-server","https-server" \
               --accelerator type=nvidia-tesla-k80,count=$num_gpus \
               --disk "name=scanner-apoms-$i,device-name=scanner-apoms-$i,mode=rw,boot=yes,auto-delete=yes" &
    done
    wait
done
