# NEURAL PAYLOAD and NEURAL EVALUATOR

Neural Payload is a Neural Network written in PyTorch for generating website attack vectors for penetration tests.
Neural Evaluator is a Neural Network for evaluating the success of a website attack.
This project is a cooperation of C&M of the Institute for Technology in Karlsruhe, Germany and IC-Consult. It was created as part of the master thesis of Cedric Mössner.


**PLEASE NOTE that this repo contains ALL the files, I used while working on the project.
There are multiple executables, helping scripts as well as networks and multiple text-files containing data or config.
Clean-Up could be done, however, I decided against it, so everyone can test for themselves, all my work can be seen and my mistakes can be comprehended and fixed. Thus, this is more to be considered, research in work in progress as a finished project.
So please use with caution!!!
However, feel free to improve, fork, create pull requests or whatever your heart (or brain) tells you to do.
Also, please refer to my thesis.**

There are many versions:
- Reinforcement Learning Neural Payload
    - Does Reinforcement Learning supervised based on a payload list
    - and unsupervised based on the attacks from a website.
- Seq2seq Neual Payload (Version 1 to 4)
    - uses a Sequence-to-Sequence Encoder-Decoder Model for Generating Payloads
    - Some Versions use an autoencoder, which did not achieve the promising results
    - Version 3 for usage, others for studies only, since they do not work
and:
- Neural Evaluator
    - uses LSTMs for evaluating. Encodes unattacked website, attacked website and payload and tries to classify which parts of the code are vulnerable. No success, only for studies.
- Neural Evaluator - Boolean Output
    - does not classify WHERE but only IF the attack was successful. Boolean output. Still uses LSTMs. No success, only for studies.
- CNN Neural Evaluator
    - uses CNNs (and differences). Works pretty well. Therefore included in the microservice API, see other project. Can be used (and studied).

# USAGE
Needs Linux, Python3 and the modules specified in requirements.txt
On Linux with python installed (default on most systems), do:
```sh
$ pip -r requirements.txt
```

then you can run it with

```sh
$ python3 script.py
```
where script.py is the script you want to execute.
Anyways, you probably need to edit the file first. Look at the section for executing under
```python
if __name__ == "__main__":
```
for how the script can be used.
