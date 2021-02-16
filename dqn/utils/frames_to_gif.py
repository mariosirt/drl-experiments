import imageio
import os

images = []
i = 0
for filename in os.listdir('./frames'):
    print(filename)
    if i >= 700:
        break
    images.append(imageio.imread('./frames/frame{}.png'.format(i+300)))
    i += 1
imageio.mimsave('../assets/example_breakout.gif', images)
