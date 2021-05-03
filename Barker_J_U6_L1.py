# Jeb Barker 5/2/2021 (based off of Mrs. Kim's shell code)
import sys, os, math


# t_funct is symbol of transfer functions: 'T1', 'T2', 'T3', or 'T4'
# input is a list of input (summation) values of the current layer
# returns a list of output values of the current layer
def transfer(t_funct, input):
    if t_funct == 'T1':
        return input
    elif t_funct == 'T2':
        return [a if a > 0 else 0 for a in input]
    elif t_funct == 'T3':
        return [1/(1+math.e ** (-1 * a)) for a in input]
    elif t_funct == 'T4':
        return [-1 + 2 / (1 + math.e ** (-1 * a)) for a in input]


# example: 4 inputs, 12 weights, and 3 stages(the number of next layer nodes)
# weights are listed like Example Set 1 #4 or on the NN_Lab1_description note
# returns a list of dot_product result. the len of the list == stage
# Challenge? one line solution is possible
def dot_product(input, weights, stage):
    return [i * w for i, w in zip(input, weights)]


# file has weights information. Read file and store weights in a list or a nested list
# input_vals is a list which includes input values from terminal
# t_funct is a string, e.g. 'T1'
# evaluate the whole network (complete the whole forward feeding)
# and return a list of output(s)
def evaluate(file, input_vals, t_funct):
    weights = []
    with open(file, "r") as f:
        x=0
        for line in f.readlines():
            for a in line.split(" "):
                try:
                    weights[x].append(float(a))
                except IndexError:
                    weights.append([float(a)])
            x+=1
    for state, w in enumerate(weights):
        if state == len(weights)-1:
            return dot_product(input_vals, w, 0)
        new = [sum(dot_product(input_vals, w[a:a+len(input_vals)], 0)) for a in range(0, len(w)-len(input_vals)+1, len(input_vals))]
        input_vals = transfer(t_funct, new) if state < len(weights)-1 else new
    return input_vals


def main():
    args = sys.argv[1:]
    file, inputs, t_funct, transfer_found = '', [], 'T1', False
    for arg in args:
        if os.path.isfile(arg):
            file = arg
        elif not transfer_found:
            t_funct, transfer_found = arg, True
        else:
            inputs.append(float(arg))
    if len(file) == 0: exit("Error: Weights file is not given")
    li = (evaluate(file, inputs, t_funct))
    for x in li:
        print(x, end=' ')


if __name__ == '__main__': main()