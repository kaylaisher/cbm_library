from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def write_parameters_tensorboard(tb_writer: SummaryWriter, parameter_dict, test_accuracy, sparsity):
    tb_writer.add_scalar("summary/test_accuracy", test_accuracy)
    tb_writer.add_scalar("summary/sparsity", sparsity)

def display_top_activated_images(*a,**k):
    fig=plt.figure(); plt.text(0.5,0.5,"viz stub",ha="center"); plt.axis("off")
    return fig
