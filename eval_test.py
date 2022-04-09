# -*- coding: utf-8 -*-
"""
Emotion Classification
Evaluation

Group 7, Simone Beckmann, Julia Evans
"""

def evaluation(corpus):
    # corpus: list of lists [[pred_label, gold_label]]
    
    gold = [ ]
    pred = [ ]
    
    # create lists of all gold and pred labels
    for entry in corpus:
        gold.append(entry[1])
        pred.append(entry[0])
    labels=[] # contains all labels (what emotions we have)
    for emotion in gold:
        if emotion not in labels:
            labels.append(emotion)
    labels = ['joy', 'guilt', 'sadness', 'shame', 'fear', 'anger', 'disgust']

    # set variables for evaluation measures
    #sum_acc = 0
    sum_prec = 0
    sum_recall = 0
    sum_F = 0
    n = len(labels)
    
    micro_tp = 0
    micro_tn = 0
    micro_fn = 0
    micro_fp = 0
    
    # column names
 #   print("EMOTION\t\t"+"TP\t","TN\t","FN\t","FP\t","PREC\t","REC\t","F\t")
    print("EMOTION\t\t","F\t")

    print("_"*70,"\n")
    
    # go through each emotion and calculate all measures
    for em in labels: 
        print(em,end="\t\t")
        
        tp = 1
        tn = 0
        fp = 0
        fn = 0
        
        for i in range(len(gold)):
            if pred[i] == em: # count tp and fp
                if pred[i] == gold[i]:
                    tp += 1
                else:
                    fp += 1               
            else: # count fn and tn
                if gold[i] == em:
                    fn += 1
                else:
                    tn += 1
        
        # calculate precision, recall, and F for current emotion
        #accuracy = (tp+tn)/(tp+tn+fp+fn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        F = (2* precision*recall)/(precision+recall)
    
      #  print(str(tp),"\t",str(tn),"\t",str(fn),"\t",str(fp),"\t",str(round(precision,2)),"\t",str(round(recall,2)),"\t",str(round(F,2)),"\n")
        print(str(round(F,2)),end = "\t\t")
 
        # add to sum (for macro-average)
        #sum_acc += accuracy
        sum_prec += precision
        sum_recall += recall
        sum_F += F
        
        # add to sum for micro-average
        micro_tp += tp
        micro_tn += tn
        micro_fn += fn
        micro_fp += fp
        
        
    # calculate and print macro and micro evaluation measures   
    print("\n"+"macro", end="")
    #print('acc:',sum_acc/n)
    print("\t"*6,round(sum_prec/n,2), end="\t ")
    print(round(sum_recall/n,2), end="\t ")
    print(round(sum_F/n,2))
                    
    micro_precision = micro_tp/(micro_tp+micro_fp)
    micro_recall = micro_tp/(micro_tp+micro_fn)
    micro_F = (2* micro_precision*micro_recall)/(micro_precision+micro_recall)         
                
    print("micro", end="")
    print("\t"*6,round(micro_precision,2), end="\t ")
    print(round(micro_recall,2), end="\t ")
    print(round(micro_F,2))

#gold_file = "isear-val2.csv"


        