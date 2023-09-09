#! bin bash -l

dir="sbatch_log"
job_File="sbatch_run.sh" 
dataset=$"fetal"
epochs=$"400"

for projector in $"8192-8192-8192" #,$"4096-4096-4096"
do
    for batch in 512 #256 512
    do 
        for lr in 0.2 #0.01 
        do 
            for version in 2 1
            do
                EXPT=barlowtwin_fetal_"$lr"_"$batch"_"$epochs"_"$version"
                STD=$dir/STD_barlowtwin_fetal_"$lr"_"$batch"_"$epochs"_"$version".out
                ERR=$dir/ERR_barlowtwin_fetal_"$lr"_"$batch"_"$epochs"_"$version".err
                export lr;
                export batch;
                export epochs;
                export version;
                export dataset;
                export projector;

                sbatch -J $EXPT -o $STD -t 00-20:00:00 -e $ERR $job_File
            done;
        done;
    done;
done;
