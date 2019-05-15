"""
Testing CSVLogger
"""


class CSVLogger(BaseLogger):
    def __init__(
        self,
        evaluator,
        log_dicts=None,
        train_log_dicts=None,
        val_log_dicts=None,
        log_dir="./logs",
        filename="logs.csv",
        **kwargs,
    ):
        """Initiate a CSV logger.


        Summary operators are created according to the parameters specified
        in the `log_dict`, `train_log_dict` and `val_log_dict` dictionaries.
        The `log_dict` dictionary contains the parameters that should be
        logged both with training and validation data, whereas the
        `train_log_dict` and the `val_log_dict` specifies the summaries that
        should be created for only the training data and validation data
        respectively. The structure of the dictionaries are as follows:
            ```
            [
                {
                    'log_name': 'Name of log 1'
                    'log_var': first_log_var'
                },
                {
                    'log_name': 'Name of log 2'
                    'log_var': 'second_log_var'
                }
            }
            ```
        The keys of the dictionaries are the name of the variables that we
        want to log. For example, if you want to log the loss of the network,
        this should the key should simply be `'loss'`. First, the evaluator
        instance is scanned for variable with the specified name (in this case,
        `loss`), then, if no variable with that name is found the network
        instance is scanned. Finally, if there is no variable with the
        specified name in the network instance the trainable parameters of the
        network is scanned.

        Below is an example of how the
        `log_dict` dictionary might look.
            ```
            [
                {
                    'log_name': 'Loss'
                    'log_var': loss'
                },
                {
                    'log_name': 'Accuracy'
                    'log_var': 'accuracy'
                }
            ]
            ```

        Parameters:
        -----------
        evaluator : utils.Evaluator
            The network evaluator to log from.
        log_dict : dict
            Logging dictionary used for both training and validation logs.
        train_log_dict: dict
            Logging dictionary used for training logs.
        val_log_dict: dict
            Logging dictionary used for validation logs.
        """
        super().__init__(
            evaluator=evaluator,
            log_dicts=log_dicts,
            train_log_dicts=train_log_dicts,
            val_log_dicts=val_log_dicts,
        )

        self.log_dir = Path(log_dir) / self.network.name
        self.filename = filename
        self.filepath = self.log_dir / filename
        self._init_logfile()
        both_summary_ops = self._init_logs(self.log_dicts)
        self.train_summary_op = self._join_summaries(
            self._init_logs(self.train_log_dicts), both_summary_ops
        )
        self.val_summary_op = self._join_summaries(
            self._init_logs(self.val_log_dicts), both_summary_ops
        )


    def _init_logfile(self):
        """Initiate an empty dataframe with the correct clumns to write the logs in.
        """
        if not self.log_dir.is_dir():
            self.log_dir.mkdir(parents=True)

        self.logs = pd.DataFrame(columns=['train', 'val', 'var_name'])

    def _join_summaries(self, *args):
        """Join the summaries to one summary list with one dict.

        The input is a series of lists containing one dictionary,
        and the output is a single list with one element which is a joined
        version of all input dictionaries.
        """
        return dict(ChainMap(*args))

    def _init_logs(self, log_dict):
        """Initiate the logging operators specified in `log_dicts`.

        The logging operator is a single dictionary with variable name as keys
        and the corresponding tensorflow operators as values.

        Parameters:
        -----------
        log_dicts : list
            List of dictionaries specifying the kind of logs to create.
            See `__init__` docstring for examples.

        Returns:
        --------
        dict : The logging operator
        """
        logs = tuple(super()._init_logs(log_dict))
        return dict(ChainMap(*logs))

    def _init_log(self, log_var, var_name, *args, **kwargs):
        """Create a specific log operator.
        
        `*args` and `**kwargs` are ignored.

        Attributes
        ----------
        log_var : tensorflow.Tensor
        var_name : str
        """
        #self.logs = self.logs.append({'train': 33, 'val': 23}, ignore_index=True)

        # add possible variable name (log operator)
        self.logs = self.logs.append({'var_name':var_name}, ignore_index=True)

        return {var_name: log_var}

    def _log(self, summaries, it_num, log_type):
        """Logs a single time step.
        """

        #her stoppet den opp:
        self.logs = self.logs.set_index('var_name')

        for name, s in summaries.items():
            self.logs[log_type].loc[name] = np.mean(s)

        # save the dataframe as a csv-file
        self.logs.to_csv(self.filepath, sep='\t', encoding='utf-8')
