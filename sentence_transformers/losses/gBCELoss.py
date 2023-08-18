import torch

# Define the sigmoid function


# Define the Generalised Binary Cross Entropy (gBCE) loss
def gBCE(predicted_scores, true_probabilities, t):
     
     
     print("predicted_scores", predicted_scores)
     print("true_probs", true_probabilities)
     predicted_probs=torch.sigmoid(predicted_scores)
     eps= 1e-5
     #predicted_probs=torch.clamp(predicted_probs,min=eps, max=1-eps)
     positive_part=true_probabilities*(torch.log(torch.pow(predicted_probs,t)))
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

     

     beta = 0.5
     print(gBCE(predicted_scores,true_probabilities,beta))