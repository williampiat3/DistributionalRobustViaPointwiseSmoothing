# Distributionnaly robust training of deep neural neural networks by using smoothed robust pointwise training
This repository contains the code for the paper "Distributionally Robust Optimization via Regularized Robust Optimization"

The dependencies are described in the file ```requirements.txt```
The robust training described in the paper is in the file "pointwise_avila.py" that should be executed by specifying some parameter:

``` python pointwise_avila.py -t <training> -r <radius> -d <device> -s <savefolder> -i <id> ```

Parameters:
* training: One of the following: ["regular","robust","adversarial","langevin"]. Type of training. The first three methods are the ones compared in the article.
* radius: positive float taken between 0.01 and 0.3 in the article. Robustness radius for robust, adversarial and lagevin trainings. This argument has no effect on the regular training.
* device: One of the following: ["cuda","cpu"]. Device where to run the training. On robust trainings prefer using the cuda device for faster trainings.
* savefolder: name of the folder where to save the model the name of the model will be "eps_<radius>_trial_<id>.pt" make sure you save different trainings with different ids or in different folders (we advise the latter).
* id: integer to identify the run, in the article we ran multiple trials (51 to be accurate) for each configuration of training and radius thus needed an id to differentiate them.

We would like to emphasize that the robust training can be computationnaly expensive (especially for large radii), a considerable part of the work not presented here was dedicated to parallelise the trainings of the differents models on a cluster in order to finish the computation in a reasonnable amount of time. The parametrization of the robust training is meant to fit on cluster GPUs, not a laptop. Reducing parameters like the sampling allows to use smaller GPUs but will degrade the robustness of the resulting models.

For statistic relevance, we performed multiple training configurations with 51 repetitions of each configuration. In order to compute the metrics on the trained models we used the "exploit_avila.py" script that computes and stores the results (it reduces the list of saved models into a result file). It has to be executed separatly on models for each trainings


The curves in the article were made upon the results of this reduction by using the "plot_pointwise.py" script.






