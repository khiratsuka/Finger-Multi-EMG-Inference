import csv
import os
import datetime

import training

def main():
    start_time = datetime.datetime.now()
    filename = os.path.join('learning_curve_csv', '20221222_terminal.txt')

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    data = {}

    with open(filename, 'r') as f:
        datalist = f.readlines()
    for i in datalist:
        if 'train_loss : ' in i :
            train_loss.append(float(i[len('train_loss : '):]))
        elif 'train_acc : ' in i :
            train_acc.append(float(i[len('train_acc : '):]))
        elif 'val_loss : ' in i :
            val_loss.append(float(i[len('val_loss : '):]))
        elif 'val_acc : ' in i :
            val_acc.append(float(i[len('val_acc : '):]))
    
    data['train_loss'] = train_loss
    data['train_acc'] = train_acc
    data['val_loss'] = val_loss
    data['val_acc'] = val_acc
    training.outputLearningCurveValue(data, start_time)
    #training.output_learningcurve(data, ['loss', 'acc'])

if __name__ == '__main__':
    main()