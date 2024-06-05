from benchopt import BaseSolver, safe_import_context
from benchopt.utils import profile
from benchmark_utils.utils import choose_denoiser
from benchopt.stopping_criterion import SingleRunCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch
    import deepinv as dinv 


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'PnP'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'denoiser_name' : ['waveletdenoiser'],
        'iteration' : ['HQS'],
        'sigma_denoiser' : [0.1],
        'stepsize' : [1.],
    }

    stopping_criterion = SingleRunCriterion()

    def set_objective(self, dataloader, physics):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.physics = physics 
        self.prior = dinv.optim.prior.PnP(denoiser = choose_denoiser(self.denoiser_name))
        self.data_fidelity = dinv.optim.data_fidelity.L2()
        self.dataloader = dataloader


    @profile
    def run(self, n_iter, plot_results=False):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.

        model = dinv.optim.optim_builder(
            iteration=self.iteration,
            prior=self.prior,
            data_fidelity=self.data_fidelity,
            early_stop=False,
            max_iter=n_iter,
            verbose=False,
            params_algo={'g_param': self.sigma_denoiser, 'stepsize': self.stepsize}
        )


        self.metric_avg, self.metric_std, _, _ = dinv.test(
            model=model,
            test_dataloader=self.dataloader,
            online_measurements=True,
            physics=self.physics,
            device= "cpu" if not torch.cuda.is_available() else "cuda",
            plot_images=plot_results,
            plot_metrics=False,
            verbose=False
        )

    def get_result(self):
        return {'metric_avg' : self.metric_avg} 