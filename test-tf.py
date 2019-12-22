import loader
import config
import train_tool
from model.pixel import Generator
import tensorflow as tf
# c=config.ConfigLoader()
# c.image_width=256
# c.image_height=256
# image_loader=loader.ImageLoader("./data/cityscapes/train", is_training=True)
# d=image_loader.load(c)

# generator=Generator()

# for x,y in d:
#     o=generator(x, training=False)
#     train_tool.save_images([x,y,o],["x","y","o"],"ex",seq=1)

#     break

# print(train_tool.parse_last_epoch("log.txt"))

save = tf.train.Checkpoint()
# save.listed = [tf.Variable(1.), tf.Variable(2.)]
# save.mapped = {'one': save.listed[0],"two":save.listed[2]}

# save.mapped = {'one': tf.Variable(1.99),"two":tf.Variable(2.44)}
# save_path = save.save('./x/tf_list_example')
# print(save_path)

# restore = tf.train.Checkpoint()
# v2 = tf.Variable(0.)
# v1=tf.Variable(0.)

# restore.mapped = {'two': v2, "one":v1}
# restore.restore(save_path)
# print(v2)
# print(v1)
ckpt=tf.train.get_checkpoint_state('./x')
print(ckpt.model_checkpoint_path)



