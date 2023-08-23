import torch

# Define the sigmoid function


# Define the Generalised Binary Cross Entropy (gBCE) loss
def gBCE(predicted_scores, true_probabilities, beta,):
     
     
     print("predicted_scores", predicted_scores)
     print("true_probs", true_probabilities)
     predicted_probs=(predicted_scores + 1)/2
     max_positive_probs = torch.max(predicted_probs * true_probabilities, dim=1, keepdim=True)[0]
     max_negative_probs = torch.max(predicted_probs * (1-true_probabilities), dim = 1, keepdim = True)[0]
     hits = torch.sum(max_positive_probs > max_negative_probs).item()
     acc =  hits/len(true_probabilities)
     print('acc', acc)
     eps= 1e-5
     #predicted_probs=torch.clamp(predicted_probs,min=eps, max=1-eps)
     positive_part=true_probabilities*(torch.log(torch.pow(predicted_probs,beta)))
     negative_part=(1-true_probabilities)*(torch.log(1-predicted_probs))
     # sigmoid_beta = sigmoid(beta * scores[0])  # for the positive example
     # neg_scores = [sigmoid(scores[i]) for i in negative_indices]  # for the negative examples
     print("predicted_probs", predicted_probs)
     print("gBCE positive", positive_part)
     print("gBCE negAtive", negative_part)
     return - (positive_part + negative_part)


if __name__ == "__main__":
     predicted_scores = torch.tensor(

     [

          [0.1, 0.2, 0.3, 0.4], 

          [0.5, 0.6, 0.7, 0.8]

     ])

     

     true_probabilities = torch.tensor( 

     

     [

          [1.0, 0.0, 0.0, 0.0], 

          [0.0, 0.0, 1.0, 0.0]

     ])

     

     t = 0.5
     print(gBCE(predicted_scores,true_probabilities,t))