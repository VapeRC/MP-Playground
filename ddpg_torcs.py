import json

import numpy as np
import tensorflow as tf

from ddpg.ActorNetwork import ActorNetwork
from ddpg.CriticNetwork import CriticNetwork
from ddpg.OU import OU
from ddpg.ReplayBuffer import ReplayBuffer
from torcs.gym import TorcsEnv


np.random.seed(1337)

BATCH_SIZE = 32
EXPLORE = 100000.

action_dim = 1 #3  # Steering/Acceleration/Brake
state_dim = 2 #29  # of sensors input


def create_networks():
    TAU = 0.001  # Target Network HyperParameters
    LRA = 0.0001  # Learning rate for Actor
    LRC = 0.001  # Lerning rate for Critic


    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)

    # Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("data/actormodel.h5")
        critic.model.load_weights("data/criticmodel.h5")
        actor.target_model.load_weights("data/actormodel.h5")
        critic.target_model.load_weights("data/criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    return actor,critic


def viz(actor, critic, mode='actor'):
    angles = np.linspace(-1, 1, 20)
    dists = np.linspace(-1, 1, 20)
    vals = np.zeros(shape=(20, 20))
    for i, angle in enumerate(angles):
        for j, dist in enumerate(dists):
            state = np.stack((angle, dist))
            state = state.reshape(1, state.shape[0])
            action = actor.model.predict(state)
            if mode == 'actor':
                val = action
            else:
                val = critic.model.predict([state, action])
            vals[i][j] = val
    angles, dists = np.meshgrid(angles, dists)

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('angle')
    ax.set_ylabel('dist')
    surf = ax.plot_surface(angles, dists, vals, cmap=cm.coolwarm)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def viz_noise(n_samples = 1000, iter = 0.):
    actions = np.linspace(-1, 1, 20)
    epsilon = 1 - iter / EXPLORE
    ou = OU()
    noise = np.zeros_like(actions)
    for i in range(n_samples):
        noise += epsilon * ou.function(actions, 0.0, 0.60, 0.30)
    noise /= n_samples
    import matplotlib.pyplot as plt
    plt.plot(actions, noise)
    plt.show()


def playGame(actor, critic, train=False):
    BUFFER_SIZE = 100000
    GAMMA = 0.99

    vision = False

    episode_count = 2000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0


    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
    ou = OU()  # Ornstein-Uhlenbeck Process

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=False, gear_change=False)

    print("TORCS Experiment Start.")
    for n_episode in range(episode_count):

        print("Episode : " + str(n_episode) + " Replay Buffer " + str(buff.count()))

        ob = env.reset()

        s_t = np.hstack((ob.angle, ob.trackPos))# ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        total_reward = 0.
        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))

            noise_t[0][0] = train * max(epsilon, 0) * ou.function(a_t_original[0][0], 0.0, 0.60, 0.30)
            #noise_t[0][1] = train * max(epsilon, 0) * ou.function(a_t_original[0][1], 0.5, 1.00, 0.10)
            #noise_t[0][2] = train * max(epsilon, 0) * ou.function(a_t_original[0][2], -0.1, 1.00, 0.05)

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            for i in range(action_dim):
                a_t[0][i] = a_t_original[0][i] + noise_t[0][i]

            ob, r_t, done, info = env.step(a_t[0])

            s_t1 = np.hstack((ob.angle, ob.trackPos))#, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            
            # Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
           
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train):
                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
        
            print("Episode", n_episode, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
        
            step += 1
            if done:
                break

        if np.mod(n_episode, 100) == 0:
            if (train):
                print("Now we save model")
                actor.model.save_weights("data/actormodel.h5", overwrite=True)
                with open("data/ctormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("data/criticmodel.h5", overwrite=True)
                with open("data/criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(n_episode) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Solvign torcs using DDPG')
    sp = parser.add_subparsers()
    sp_train = sp.add_parser('train', help='Train agent')
    sp_train.set_defaults(mode='train')
    sp_run = sp.add_parser('run', help='Evauluate agent')
    sp_run.set_defaults(mode='run')
    sp_viz = sp.add_parser('viz', help='Visualize critic and agent')
    sp_viz.set_defaults(mode='viz')
    sp_viz.add_argument('func', choices=('actor', 'critic', 'noise'))
    args = parser.parse_args()
    actor, critic = create_networks()
    if args.mode == 'train':
        playGame(actor, critic, train=True)
    elif args.mode == 'run':
        playGame(actor, critic, train=False)
    elif args.mode == 'viz':
        if args.func == 'noise':
            viz_noise()
        else:
            viz(actor, critic, mode=args.func)