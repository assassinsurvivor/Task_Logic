### Task_Logic
Assignment

# To run pretraining code:
        python model_train_language.py --en 4 --ed 150 --hd 300 --ah 3 --dr 0.1 --ep 500 --dm 50
    
# Fine-tuning code:
        python model_train.py --en 4 --ed 150 --hd 300 --ah 3 --dr 0.1 --ep 50
        
1. To upload a directory containing files as a bundle:
        
        Client.create_and_upload_bundle(__dir_path__, tag="file_bundle_upload")

1. To set up resources :
        
        res=sc.Resources(cpu=4,gpu=1,memory=16,disk=32,gpu_memory=__['8GB','16GB', '32GB']__) #To choose from available resources we can pass a list

1. To add bundles and smi to the instance :
        
        req=sc.Requirements(bundles=__[bid]__,smi='ubuntu18.04-cuda10.2-410.79')
    
1. To install specific python libraries along with a specific python package we can add that command as a part of the job task:
        
        command_python=sc.Command(["/bin/sh", "-c", "curl https://bootstrap.pypa.io/ez_setup.py -o - | python3.6 && python3.6 -m easy_install pip"])
        command_lib=sc.Command(["/bin/sh", "-c", "pip install pandas numpy "])
        job = sc.Job(tasks=[command_python,command_lib],resources=res ,requirements=req description="running_first_job")
    
1. To submit a job :
        
        Client.submit_job(job)
    
1. To download output files created inside a job first we define the output structure and add that to a bundle which can be downloaded:
        
        simcloud.Output(__[list_of_output_files_to_download]__) #staging area
        Client.add_output_bundle_task(job,outputs=__[list_of_output_files_to_download]__)
        Client.download_output(job=job_info,dest_dir=__[local_save_path]__)
    
1. To delete a bundle or a job after successful completion :

        Client.archive_job(__job-id__)
        Client.delete_bundle(__bundle-id__)
        
    
