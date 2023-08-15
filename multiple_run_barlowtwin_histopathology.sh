#! bin bash -l

dir="sbatch_log"
job_File="sbatch_run.sh" 
dataset=$"histopathology"
epochs=$"100"

for projector in $"8192-8192-8192"
do
    for batch in 512
    do 
        for lr in 0.2
        do 
            for version in 1 2
            do
                EXPT=barlowtwin_histopathology_"$lr"_"$batch"_"$epochs"_"$version"
                STD=$dir/STD_barlowtwin_histopathology_"$lr"_"$batch"_"$epochs"_"$version".out
                ERR=$dir/ERR_barlowtwin_histopathology_"$lr"_"$batch"_"$epochs"_"$version".err
                export lr;
                export batch;
                export epochs;
                export version;
                export dataset;
                sbatch -J $EXPT -o $STD -t 01-23:00:00 -e $ERR $job_File
            done;
        done;
    done;
done;
