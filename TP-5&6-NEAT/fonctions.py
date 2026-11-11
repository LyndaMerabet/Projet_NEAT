from poles_physics import SinglePoleEnv
from neat_classes import *
from neat_evolution import *

LIMITE_PISTE = 2.4
FORCE_MIN = 10.0
FORCE_MAX = 40.0
FORCE_X  = 20.0  
DURATION = 10.0

initial_state = [0.0, 0.0, np.pi + 0.05, -0.5]

POPULATION_SIZE = 100
NUM_EPOCHS = 50
SURVIVAL_RATE = 0.5 
KEEP_BEST =  2  

def angle_error(theta):
    return (theta- np.pi + np.pi) % (2.0 * np.pi) - np.pi
    
def better_naive_single_controller(observation):
    x = observation[0]
    x_dot = observation[1]
    theta = observation[2]
    theta_dot = observation[3]

    if theta_dot == 0.0 and theta == np.pi:
        return 0.0
    
    angle_error_val = angle_error(theta)
    
    if angle_error_val > 0.0:             #coté gauche
        if theta_dot < 0.0: 
            force = -FORCE_MIN               #il se redresse on pousse un peu
        else:                         
            force = -FORCE_MAX                 #il tombe donc on pousse fort
    else:                            #coté droit
        if theta_dot > 0.0:           #il se redresse
            force = FORCE_MIN
        else:                        #il tombe
            force = FORCE_MAX

    #maintenant on regarde la position du chariot 
    if x+x_dot < -LIMITE_PISTE:  #il est trop à gauche
        force -= FORCE_X
    elif x+x_dot > LIMITE_PISTE: #il est trop à droite
        force += FORCE_X
    
    return force

def better_naive_double_controller(observation):
    x, x_dot, theta1, theta_dot1, theta2, theta_dot2 = observation

    error_1 = (theta1 - np.pi + np.pi) % (2.0 * np.pi) - np.pi
    error_2 = (theta2 - np.pi + np.pi) % (2.0 * np.pi) - np.pi

    force = -( 1 * x + 4 * x_dot
             + 400 * error_1 + 40 * theta_dot1
             - 400 * error_2 - 100 * theta_dot2)

    return float(np.clip(force, -50, 50))

##############################################################################


genome = create_initial_genome(
    genome_id=0,
    num_inputs=4,
    num_outputs=1,
    fully_connected=True
)
#visualize_genome(genome, title="Réseau initial 4→1")
#output = genome.forward_pass(initial_state)
#print(output[0])
    
def evaluate_pendule(genome: Genome) :
    env = SinglePoleEnv(initial_state=initial_state)
    observation = env.reset()
    pas = int (DURATION / env.dt)
    resultat = 0.0
    
    for i in range(pas):
        force = genome.forward_pass([observation[0], observation[1], angle_error(observation[2]), observation[3]])[0]
        # force avec la sigmoide sera entre 0 et 1 or la force est entre -50 et 50 donc avec
        # le -0.5 on se met entre -0.5  et +0.5 puis * 100 
        force = (force - 0.5 ) * 100.0
        observation, terminated = env.step(force)
        angle = angle_error(observation[2]) #on revérifie l'angle résultant
        if terminated or abs(angle) > np.pi/2:
            break
        resultat += 1.0
    env.close()
    return (resultat * env.dt)/10.0  # Normalisation du score entre 0 et 1


#creer les 100 reseaux de depart
history = evolve_population(
    evaluate_fn=evaluate_pendule,
    num_inputs=4,
    num_outputs=1,
    population_size=POPULATION_SIZE,
    num_epochs=NUM_EPOCHS,
    survival_rate=SURVIVAL_RATE,
    keep_best=KEEP_BEST,
    crossover_prob=0.2,
    random_seed=42,
    verbose=True,
    verbose_interval=5,
    use_species=True,  # ACTIVER LA SPÉCIATION
    c1=1.0,
    c2=1.0,
    c3=0.4,
    compatibility_threshold=3.0,
    early_stop_fitness=0.999
)

# if history is not None:
#     plot_evolution_results(
#         history=history,
#         population_size=POPULATION_SIZE,
#         log_scale=False
#     )

for seed in (0, 1, 2):
    h = evolve_population(evaluate_fn=evaluate_pendule, num_inputs=4, num_outputs=1,
                           population_size=100, num_epochs=30, survival_rate=0.5,
                              keep_best=2, crossover_prob=0.2, use_species=True,
                              random_seed=seed, verbose=False, early_stop_fitness=0.999)
    print( seed, h['best_genome'], h['best_genome'].fitness)