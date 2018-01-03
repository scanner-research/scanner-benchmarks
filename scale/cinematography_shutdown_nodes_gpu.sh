if [ -z "$1" ]
  then
    echo "Usage: cinematography_spawn_nodes.sh <start_node> <num_nodes>"
    exit
fi

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

start_node=$1
num_nodes=$2
num_simul=100

max=$(($num_nodes + $num_simul - 1))
for j in `seq $start_node $num_simul $max`; do
    end=$(($j + $num_simul))
    m=$(( $num_nodes < $end ? $num_nodes : $end ))
    echo "Destroying nodes $j through $m..."
    for i in `seq $j 1 $m`; do
        yes | gcloud compute --project "visualdb-1046" \
                     instances delete "scanner-apoms-$i" \
                     --zone "us-east1-d" \
                     --delete-disks=all &
    done
    wait
done
