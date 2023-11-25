''' this code is the implementation of "Strong transitivity relations and Graph neural network" paper

For the graph neural network layer, PYG lib 
 	installation link : https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html


'''

For run:
         "python run.py"


For run with desire parameter and dataset run:
        dataset : "cora", "citeseer","pubmem","USA","actor","twitch"
	hidden	: hidden dimension
	th 	: threshold for transitivity graph construction
        epochs	: number of epochs

	"python run.py --dataset USA --hidden 16 --th 0.85"
	"python run.py --dataset USA --hidden 16 --th 0.85 --epochs 100"



