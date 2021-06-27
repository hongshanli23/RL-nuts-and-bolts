

def get_actor_grads(self, normalized_obs0):
    with tf.GradientTape() as tape:
        actor_tf = self.actor(normalized_obs0)
        normalized_critic_with_actor_tf = self.critic(normalized_obs0, actor_tf)
        critic_with_actor_tf = denormalize(tf.clip_by_value(normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        actor_loss = -tf.reduce_mean(critic_with_actor_tf)
    actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
    if self.clip_norm:
        actor_grads = [tf.clip_by_norm(grad, clip_norm=self.clip_norm) for grad in actor_grads]
    if MPI is not None:
        actor_grads = tf.concat([tf.reshape(g, (-1,)) for g in actor_grads], axis=0)
    return actor_grads, actor_loss

def get_critic_grads(self, normalized_obs0, actions, target_Q):
    with tf.GradientTape() as tape:
        normalized_critic_tf = self.critic(
            normalized_obs0, actions)
        normalized_critic_target_tf = tf.clip_by_value(
            normalize(target_Q, self.ret_rms), 
            self.return_range[0], self.return_range[1])
        critic_loss = tf.reduce_mean(tf.square(
            normalized_critic_tf - normalized_critic_target_tf))
        
        if self.critic_l2_reg > 0:
            for layer in self.critic.network_builder.layers[1:]:
                critic_loss += (self.critic_l2_reg/2.0) * \
                tf.reduce_sum(
                    tf.square(layer.kernel))
    
    critic_grads = tape.gradient(critic_loss, 
        self.critic.trainable_variables)
    if self.clip_norm:
        critic_grads = [tf.clip_by_norm(grad, clip_norm=self.clip_norm)
            for grad in critic_grads]
    
    # flatten the grad and concatinate 
    # grad in each layer into a long vector
    # for MPI 
    if MPI is not None:
        critic_grads = tf.concat(
            [tf.reshape(g, (-1,)) for g in critic_grads],
            axis=0)
    return critic_grads, critic_loss

# Looks like every tensor involved in computing gradient
# is normalized 

def normalize(x, stats):
    """
    normalize x according to historic mean and standard 
    deviation

    stats: historic mean and standard deviation
    x: tensor to normalize
    """
    if stats is None:
        return x
    return (x - stats.mean)/(stats.std + 1e-8)



