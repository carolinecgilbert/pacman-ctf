"""
- Suicide / deception
- go back home after threshold of pills: percentage threshold (65 base)
- If ghost is scared run away from pacman
- universal class for switching agent behaviour
- if one dies, switch
- Higher level behaviour tree, lower level decisions expectimax
- if initial position then request switch
- return to homebase (other team agent) function
- if being chased then go home or go to capsule (whatever is closest)
Behaviour tree pacman: 
      - calculate what is the hardest food to get and only go for that food if you have the pill
      - 40* moves after pill so find opitmal path in 40 moves that gets most pills


WEEK 20 MEETING NOTES
- Use geometric mean to normalize heuristic function
- Check alpha-beta tree
- Check that legal actions are correct
- Add heuristic that rewards a smaller ghost list
"""
