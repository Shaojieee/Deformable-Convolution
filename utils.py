import pandas as pd
import torch
import numpy as np

def evaluation_fn(y_true, y_pred, **kwargs):
    return {
        'Accuracy': torch.sum(y_true==torch.argmax(y_pred, axis=1)).item()/len(y_true),
        'Avg Cross Entropy Loss': torch.nn.functional.cross_entropy(y_pred, y_true).item(),
    }
    


# Runs the evaluation_fn and store and prints the results.
class EvaluationCallback():
    def __init__(self, evaluation_fn, type):
        self.evaluation_fn = evaluation_fn
        self.type = type
        self.logs = []

    def on_epoch_end(self, epoch, Y_true, Y_pred, time_taken, **kwargs):
        
        results = self.evaluation_fn(Y_true, Y_pred, **kwargs)

        output_string = f'{self.type} Metrics: Epoch: {epoch}'
        for k, v in results.items():
            output_string += f' {k}: {v:.6f}'

        output_string += f' Time Taken: {time_taken:.3f}'
        
        print(output_string)
        results['epoch'] = epoch
        results['time_taken'] = time_taken

        self.logs.append(results)
    
        return False
                   
    def get_logs(self):
        return pd.DataFrame(self.logs)
    
    def save_results(self, output_dir, file_name):
        results = pd.DataFrame(self.logs)
        results.to_csv(f'{output_dir}/{file_name}')


# Save models best weights and also provide early stopping functionality
class ModelCheckpoint():
    def __init__(self, early_stop=False, patience=5, min_delta=0, mode='max', restore_best_weights=False):
        # Input Parameters
        self.early_stop = early_stop # Whether to do early stopping check
        self.patience = patience # Patience for early stopping
        self.min_delta = min_delta # Min Delta for early stopping
        self.mode = mode # 'min' if lower is better, 'max' if higher is better
        self.restore_best_weights = restore_best_weights # Whether to restore the best weights after completing training
        
    
        self.best_weights = None # The models best weights
        self.best = np.inf if mode=='min' else -np.inf # The models best loss for early stopping
        self.stopped_epoch = 0 # Epoch where model is best for early stopping
        self.wait = 0 # Early stop check
        
        if mode=='min':
            self.monitor_op = np.less
        elif mode=='max':
            self.monitor_op = np.greater
              
    def on_epoch_end(self, model, loss, epoch, **kwargs):
        
        if self.monitor_op(loss - self.min_delta, self.best):
            self.best = loss
            self.wait = 0
            model.eval()
            self.best_weights = model.state_dict()
        elif self.early_stop:
            self.wait += 1
            
            if self.wait>=self.patience:
                self.stopped_epoch = epoch
                print(f'Early Stopping at Epoch {epoch} with best loss of {self.best:.6f}')
                if self.restore_best_weights:
                      print(f'Restoring best weights at Epoch {self.stopped_epoch-self.wait}')
                      model.eval()
                      model.load_state_dict(self.best_weights)
                return True
        return False
    
    def save_best_weights(self, accelerator, output_dir, file_name=None):
        if file_name==None:
            file_name = 'best_weights.pth'
        accelerator.save(self.best_weights, f'{output_dir}/{file_name}')
    
    def load_best_weights(self, model):
        if self.restore_best_weights:
            print(f'Restoring best weights at Epoch {self.stopped_epoch-self.wait}')
            model.eval()
            model.load_state_dict(self.best_weights)
