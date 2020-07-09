### Task_Logic
Assignment

# To run pretraining code:
        python model_train_language.py --en 4 --ed 150 --hd 300 --ah 3 --dr 0.1 --ep 500 --dm 50
    
# Fine-tuning code:
        python model_train.py --en 4 --ed 150 --hd 300 --ah 3 --dr 0.1 --ep 50


## To run job:
    -python post_single_job_sim.py --gpu 1 --cpu 4 --of data.csv --pfp ./files/etl.py --fp./files/ --savedir results/

<br />

In case if we want to install library on top of a bundle we can just add a simcloud command and add that to a task job.
    -sc.Command(["/simcloud-packages/venv/bin/pip","install","mysql-connector"])
