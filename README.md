# udactity-drlnd-projects
Repository of Udacity Deep Reinforcement Learning Nanodegree Program project code.

## Project details

### The "Bananas Collector" environment

![](./assets/banana-collector.gif)

#### State space

The state space has 37 dimensions and contains the agent's velocity, along with ray-based 
perception of objects around agent's forward direction. 

#### Action space

The simulation contains a single agent that navigates a large environment. At each time step, the 
agent has four actions at its disposal:

0. Walk forward
1. Walk backward
2. Turn left
3. Turn right

#### Rewards 

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for 
collecting a blue banana. The goal of the agent is to collect as many yellow bananas as possible 
while avoiding blue bananas.The task is episodic, and in order to solve the environment, the agent 
must get an average score of +13 over 100 consecutive episodes.

### Downloading the environment

The specific version of the "Bananas Collector" Unity environment used in this project is 
available at the links below.

* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip) 
* [Linux (headless)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip)
* [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

After downloading the appropriate environment file move the file to the `./environments` directory 
in this repository and the unzip the archive. After unzipping the archive you can then determine 
the path to the environment for your OS.

* Linux: `./environments/Banana_Linux/Banana.x86_64`
* Linux: `./environments/Banana_Linux_NoVis/Banana.x86_64`
* Mac OSX: `./environments/Banana.app`

## Getting started

### Building the Conda environment

I use [Conda](https://docs.conda.io/en/latest/) to manage the environment for this project. Create 
the Conda environment in a sub-directory `./env`of the project directory by running the following 
commands.

```bash
export ENV_PREFIX=$PWD/env
conda env create --prefix $ENV_PREFIX --file environment.yml --force
```

Once the new Conda environment has been created you can activate the environment with the following 
command.

```bash
conda activate $ENV_PREFIX
```

If you wish to use any JupyterLab extensions included in the `environment.yml` and `requirements.txt` 
files then you need to activate the environment and rebuild the JupyterLab application using the 
following commands to source the `postBuild` script.

```bash
conda activate $ENV_PREFIX # optional if environment already active
. postBuild
```

For your convenience these commands have been combined in a shell script `./bin/create-conda-env.sh`. 
Running the shell script will create the Conda environment, activate the Conda environment, and 
build JupyterLab with any additional extensions. The script should be run from the project root 
directory as follows.

```bash
./bin/create-conda-env.sh
```

## Instructions

Interactive training can be done using JupyterLab. Run the following commands to launch the 
JupyterLab server

```bash
conda activate $ENV_PREFIX
jupyter lab
```

and then open and run the contents of `notebooks/Navigation.ipynb`. Depending on your OS you may 
need to modify the path to your environment in the notebook.
