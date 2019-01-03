import tensorflow as tf

flags = tf.flags

flags.DEFINE_string("mode", "train", "Running mode train/debug/test")

def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        print('this is train')
def test(_):
    print('this is test')

if __name__ == "__main__":
    # 主函数中的tf.app.run()会调用main，并传递参数，因此必须在main函数中设置一个参数位置。
    # 如果要更换main名字，只需要在tf.app.run( )中传入一个指定的函数名即可。
    # default肯定是main啦
    tf.app.run(test)
