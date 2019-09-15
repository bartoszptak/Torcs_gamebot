from gym_torcs.gym_torcs import TorcsEnv
from gamebot import Torcs as gamebot
from model import get_model
import numpy as np
import cv2

vision = True


def main(train=False, eval=False):
    # Generate a Torcs environment
    game = TorcsEnv(vision=vision, throttle=True, gear_change=False)

    bot = gamebot(get_model(), train)
    reset_counter = 0

    next_state, reward, done = game.reset(relaunch=True), 0, False
    next_img, next_speed = bot.preprocessing(next_state)

    img = np.concatenate((next_img, next_img, next_img, next_img), axis=3)

    speed = np.stack((next_speed, next_speed, next_speed, next_speed), axis=0)
    speed = np.reshape(speed, (1, *speed.shape))

    while True:
        action = bot.make_action(img, speed)

        next_state, reward, done, _ = game.step(action)

        next_img, next_speed = bot.preprocessing(next_state)
        next_img = np.append(next_img, img[:, :, :, :9], axis=3)
        next_speed = np.reshape(next_speed, (1, 1, *next_speed.shape))
        next_speed = np.append(next_speed, speed[:, :3], axis=1)

        if train:
            bot.make_buffer(img, speed, next_img, next_speed, reward, done)
            train_index, loss = bot.make_train()
            print(f'Epoch: {train_index} - loss: {loss}')
            if train_index == bot.EXPLORE:
                return

        if done:
            if reset_counter % 3 == 0:
                next_state = game.reset(relaunch=True)
                reset_counter = 0
            else:
                next_state = game.reset()
                reset_counter += 1

        img, speed = next_img, next_speed

    game.end()


if __name__ == "__main__":
    train = False
    eval = False
    main(train, eval)