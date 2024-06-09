# SuperMarioBrosRL
Authors: [Natalia Safiejko](https://github.com/ssafiejko), [Miłosz Kita](https://github.com/miloszkita), [Kacper Wnęk](https://github.com/KacWNK), [Wojciech Grabias](https://github.com/WojtekGrbs)
# About the project
The major goal of the project was to use Reinforcement Learning Techniques and implement them to teach an agent to beat Super Mario Bros Stage 1-1 in the shortest time possible.
<p align="center">
<img src="https://github.com/WojtekGrbs/SuperMarioBrosRL/assets/51636941/726ae507-80eb-4f4b-ab6a-582de4a986ca" width="250" height="250"/>
</p>

## Project Setup
All requirements except the `pytorch` library are included in the `requirements.txt` file:
```
pip install -r requirements.txt
```
It is recommended to install the CUDA version of `pytorch` separately, directly from the source url:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Remember to choose the accurate CUDA version. If CUDA is not available on your device, you can install the regular version of the package.
## Project Structure
- `util/`:
  -  `charts.py` responsible for creating creating learning charts
  -   `rewards.py` responsible for reward functions
  -    `preparation.py` that includes environmental variables and modifications.
- `model_eval/`
  - `model2gif.py` allows you to change a trained model into a gif
  - `play_model.py` responsible for re-playing a selected model from a checkpoint
- `agent/`
  - `mario.py` contains the implementation of DDQN algorithm as a `Mario` class
  - `smbneuralnetwork.py` contains Neural Network architecture
## Technical approach
### Environment
The project used [OpenAI's gym package](https://pypi.org/project/gym-super-mario-bros/) in order to emulate the game in Python. Core envrionment variables and wrappers are included in `util/preparation.py`. Feel free to modify them and see the output of your learning!
### Neural Networks
In order to train the agent we used the Double Deep Q Learning algorithm that involved two separate Neural Networks. The architecture of those networks is the same with one exception - target network requires no gradient calculation throught the learning process. Structure of the networks was implemented in `agent/smbneuralnetwork.py`. <br> **Changing the environment variables will probably require you to change the architecture of the networks.**
#### Input Data
The input data of the Neural Networks is a tensor of 4 black and white frames. The learning process relies only on the image of the game.
<p align="center">
<img src="https://github.com/WojtekGrbs/SuperMarioBrosRL/assets/51636941/5242eb86-c58b-4edc-a24c-5e00d951a83d" width="250" height="250"/>
</p>

### Learning process
Starting the learning process is as simple as:
```
python main.py
```
The script will automatically create a directory in which it will later save checkpoints of your learning process. At the end of that stage it will also create the final model. Switch the `RENDER_MODE` variable in `preparation.py` between `rgb_array` and `human` in order to turn the game window off/on.
### Model
#### Model replay
In order to replay a trained model via a checkpoint you need to change the `CHECKPOINT_PATH` in the `model_eval/play_model.py` file and run it through
```
python ./model_eval/play_model.py
```
#### Model gif save
In order to replay a trained model via a checkpoint you need to change the `CHECKPOINT_PATH` in the `model_eval/model2gif.py` file and run it through
```
python ./model_eval/model2gif.py
```
**NOTE: Remember to set the `FILENAME` to desired `*.gif` path in order to save the gif in a desired destination!**
### Results
#### The best achieved run of Mario was with the time of 339:
<p align="center">
<img src="https://github.com/WojtekGrbs/SuperMarioBrosRL/assets/51636941/00dc4537-e900-436e-b751-e5981c0ec0ee" width="250" height="250"/>
</p>
<p align="center">
<img src="https://github.com/WojtekGrbs/SuperMarioBrosRL/assets/51636941/9d449361-9d01-4845-b560-9eac49471f6b" width="600" height="420" />
</p>

