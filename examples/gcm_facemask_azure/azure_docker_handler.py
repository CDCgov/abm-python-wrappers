from abmwrappers import utils

super_experiment_name = "wtk-gcm-azb"

## ======================================#
## # Use inputs------
## ======================================#

# Azure batch inputs
create_pool = True
ab_config_path = "examples/azb_config.yaml"
files_to_upload = [
    "examples/gcm_facemask_azure/model_run.py",
    "examples/gcm_facemask_azure/input/config.yaml",
]

## ======================================#
## # Create/specify pool------
## ======================================#

# Question: autoscale with lots of nodes? Node type?

client, blob_container_name, job_prefix = utils.initialize_azure_client(
    ab_config_path, super_experiment_name, create_pool
)

if client and blob_container_name and job_prefix:
    print("Azure Client initialized successfully.")
    print("Blob container name:", blob_container_name)
else:
    print("Failed to initialize Azure Client.")

## ======================================#
## # Upload python script and input file------
## ======================================#

client.upload_files(files_to_upload, f"{blob_container_name}")

## ======================================#
## # Run python script------
## ======================================#

job_name = f"{job_prefix}py_wrapper"
client.add_job(job_name)

docker_command = (
    f"""python /{blob_container_name}/{files_to_upload[0].split('/')[-1]}"""
)
print("docker command:", docker_command)
client.add_task(job_name, docker_cmd=docker_command)
