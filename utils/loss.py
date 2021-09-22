import tensorflow.keras.backend as K
import tensorflow as tf

def loss():

    def dsc(y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return score

    #Dice loss according dice coefficient
    def dice_loss(y_true, y_pred):
        loss = 1 - dsc(y_true, y_pred)
        return loss

    #Jacard loss
    def jaccard_loss(output, target, axis=(0, 1, 2), smooth=1e-5): 
        inse = tf.reduce_sum(output * target, axis=axis)
        
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
        
        jaccard = 1- (inse + smooth) / (l + r - inse + smooth)
        jaccard = tf.reduce_mean(jaccard)

        return jaccard


    #Tversky index
    def tversky(y_true, y_pred):
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1-y_pred_pos))
        false_pos = K.sum((1-y_true_pos)*y_pred_pos)
        alpha = 0.7
        return (true_pos + 1)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + 1)

    #Focal loss according to tversky index
    def focal_tversky_loss(y_true,y_pred):
        pt_1 = tversky(y_true, y_pred)
        gamma = 0.75
        return K.pow((1-pt_1), gamma)


        return jaccard_loss, focal_tversky_loss, dice_loss
