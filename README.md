

## Installation

```bash
# Clone the code to local machine
git clone https://github.com/BELIV-ASU/aPVP0.9.10.1.git
cd aPVP0.9.10.1.git

# Create Conda environment
conda create -n pvp python=3.7
conda activate pvp

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install evdev package (Linux only)
pip install evdev


# You now have installed MetaDrive and MiniGrid.
# To set up CARLA dependencies, please click the details below.
```

<details>
<summary><b>Set up CARLA dependencies</b></summary>

```bash
# Step 1: Download and unzip CARLA 0.9.10.1 to your home folder
cd ~/
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz
export CARLA_ROOT="CARLA_0.9.10.1"
mkdir ${CARLA_ROOT}
tar -xf CARLA_0.9.10.1.tar.gz -C ${CARLA_ROOT}  # CARLA is stored at: ~/CARLA_0.9.10.1

# Step 2: Setup the environment variables
vim ~/.bashrc
# Add following sentences and replace PATH_TO_CARLA_ROOT with the path to ${CARLA_ROOT} 
export CARLA_ROOT="~/CARLA_0.9.10.1"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg":${PYTHONPATH}

# Step 3: Activate your conda environment and test if CARLA is installed correctly.
conda activate pvp  # If you are using conda environment "pvp"
python -c "import carla"  # If no error raises, the installation is successful.

# Step 4: Install dependencies
pip install DI-engine==0.2.2
pip install torchvision
pip install markupsafe==2.0.1

# NOTE: If you are using a new conda environment, you might need to reinstall 'pvp' repo.
# Now let's jump to the CARLA section to run experiment!
```
</details>


## Launch Experiments


### CARLA

We use CARLA 0.9.10.1 as the backend and use the environment created by [DI-Drive](https://github.com/opendilab/DI-drive) as the gym interface. CARLA uses a server-client architecture. To run experiment, launch the server first:

```bash
# Launch an independent terminal, then:
cd ~/CARLA_0.9.10.1  # Go to your CARLA root
./CarlaUE4.sh -carla-rpc-port=9000  -quality-level=Epic  # Can set to Low to accelerate
# Now you should see a pop-up window and you can use WASD to control the camera.
```

Click for the experiment details:

<details>
  <summary><b>CARLA - Steering Wheel (Logitech G29)</b></summary>

Note: Do not connect Xbox controller with the steering wheel at the same time!

```bash
# Launch the CARLA server if you haven't done yet
~/CARLA_0.9.10.1/CarlaUE4.sh -carla-rpc-port=9000  -quality-level=Epic  # Can set to Low to accelerate

# Go to the repo root
cd ~/pvp

# Run experiment without Wandb:
python pvp/experiments/carla/train_pvp_carla.py --exp_name pvp_carla_test

# Run full experiment
python pvp/experiments/metadrive/train_pvp_metadrive.py \
--exp_name pvp_carla \
--wandb \
--wandb_project WADNB_PROJECT_NAME \
--wandb_team WANDB_ENTITY_NAME
```

| Action             | Control                 |
|--------------------|-------------------------|
| Throttle           | Throttle pedal          |
| Human intervention | Left/Right gear shifter |
| Steering           | Steering wheel          |
</details>




