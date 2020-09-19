# raisimGymTrial
Trial of [raisimTech/raisimlib](https://github.com/raisimTech/raisimlib) ver 1.0.0 + [pfnet/pfrl](https://github.com/pfnet/pfrl) 

## Prerequirements
- Pytorch==1.6
- see requirements.txt

## Installation

```
git submodule update --init --recursive
export LOCAL_INSTALL=$HOME/raisim
mkdir ./raisimlib/build
cd ./raisim/build/
cmake .. -DCMAKE_INSTALL_PREFIX=$LOCAL_INSTALL -DRAISIM_EXAMPLE=ON -DRAISIM_PY=ON
cmake --build . --target install
```

## Build Gym Environment
Apply patch to make raisimGymTorch more compatible with OpenAI I/F, then build.

```
export LOCAL_BUILD=$HOME/raisim
patch -p0 < ./patch/patch.txt
cd ./raisimlib/raisimGymTorch
python setup.py develop --CMAKE_PREFIX_PATH $LOCAL_BUILD --user
```

## Run raisimUnity
To visualize Raisim simulation, Raisim unity is the recommended tool. Run binary under raisim unity in raisimlib.

## Run Training
Place the license file named `activation.raisim` under `$RAISIM_HOME`

```
export RAISIM_HOME=$LOCAL_BUILD
python train_ppo_pfrl.py --render --num-envs 3
```
