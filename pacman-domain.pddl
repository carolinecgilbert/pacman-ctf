
    (define (domain pacman)

      (:requirements
          :typing
          :negative-preconditions
      )

      (:types
          foods cells
      )

      (:predicates
          (cell ?p)

          ;pacman's cell location
          (at-pacman ?loc - cells)

          ;food cell location
          (at-food ?f - foods ?loc - cells)

          ;Indicates if a cell location has a ghost
          (has-ghost ?loc - cells)

          ;Indicated if a cell location has a capsule
          (has-capsule ?loc - cells)

          ;connects cells
          (connected ?from ?to - cells)

          ;pacman is carrying food
          (carrying-food)

          ;capsule eaten
          (capsule-eaten)

          ;want to die
          (want-to-die)

          ;die
          (die)
      )

      ; move pacman to location with no ghost
      (:action move
          :parameters (?from ?to - cells)
          :precondition (and
              (at-pacman ?from)
              (connected ?from ?to)
              (not (has-ghost ?to))
          )
          :effect (and
                      ;; add
                      (at-pacman ?to)
                      ;; del
                      (not (at-pacman ?from))
                  )
      )

      (:action move-no-restriction
          :parameters (?from ?to - cells)
          :precondition (and
              (at-pacman ?from)
              (connected ?from ?to)
              (want-to-die)
          )
          :effect (and
                      ;; add
                      (at-pacman ?to)
                      ;; del
                      (not (at-pacman ?from))
                  )
      )

      ; move pacman to food location if there no ghost
      (:action eat-food
          :parameters (?loc - cells ?f - foods)
          :precondition (and
                          (at-pacman ?loc)
                          (at-food ?f ?loc)
                        )
          :effect (and
                      ;; add
                      (carrying-food)
                      ;; del
                      (not (at-food ?f ?loc))
                  )
      )

      ; move pacman to food location if there no ghost
      (:action eat-capsule
          :parameters (?loc - cells)
          :precondition (and
                          (at-pacman ?loc)
                          (has-capsule ?loc)
                        )
          :effect (and
                      ;; add
                      (capsule-eaten)
                      ;; del
                      (not (has-capsule ?loc))
                  )
      )

      (:action move-after-capsule-eaten
          :parameters (?from ?to - cells)
          :precondition (and
              (at-pacman ?from)
              (connected ?from ?to)
              (capsule-eaten)
          )
          :effect (and
                      ;; add
                      (at-pacman ?to)
                      ;; del
                      (not (at-pacman ?from))
                  )
      )

      ; move pacman to ghost location to die
      (:action get-eaten
          :parameters (?loc - cells)
          :precondition (and
                          (at-pacman ?loc)
                          (has-ghost ?loc)
                        )
          :effect (and
                      ;; add
                      (die)
                      ;; del
                      ;; (not (has-ghost ?loc))
                  )
      )
    )
    