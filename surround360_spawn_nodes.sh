if [ -z "$1" ]
  then
    echo "Usage: surround360_spawn_nodes.sh <num_nodes>"
    exit
fi

num_nodes=$1
num_gpus=8

SSH_KEY="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC+jBMU/O4ePOmxVoodKC/dz5+G4uyz2PmPdOSBFDkZJjHSvIlj+yKHCfM8uFQLct0mSg3jHy7RZDCSIcuXX99wBoWrFwNns8ODwAULYZt6aw8Yk6tHnG4gtz6nKM0k/AxqRFXqxZ94ON8YzOxY/hXNW1syFsIrbryBK2py8xHcn86pdwuSkGvEW5TyMG9AnngnITfOzgVENJ7RE0kLaxD295jMkJiM5XfVvhrcfs1X+YHHZTA2fYLI73LmCtkBFe2M8R9BTKsyv7myhnMjv4V/RxI7I6Z/XGXWvVTfb7GyTs7Dy+RDJa3z9SG+q7jeDUPcssZ3osJjUce+To1MYqi1 apoms@Alexanders-MacBook-Pro.local"

for i in `seq 1 $num_nodes`; do
    gcloud compute --project "visualdb-1046" \
           disks create "surround360-scale-$i" \
           --size "200" \
           --zone "us-east1-d" \
           --source-snapshot "surround360-scale" \
           --type "pd-ssd" &
done

wait

for i in `seq 1 $num_nodes`; do
    gcloud beta compute --project "visualdb-1046" \
           instances create "surround360-scale-$i" \
           --zone "us-east1-d" \
           --machine-type "n1-standard-64" \
           --network "default" \
           --metadata "ssh-keys=ubuntu:$SSH_KEY,node_id=$i" \
           --maintenance-policy "TERMINATE" \
           --service-account "50518136478-compute@developer.gserviceaccount.com" \
           --scopes "https://www.googleapis.com/auth/cloud-platform" \
           --tags "http-server","https-server" \
           --disk "name=surround360-scale-$i,device-name=surround360-scale-$i,mode=rw,boot=yes,auto-delete=yes" &
done

wait

