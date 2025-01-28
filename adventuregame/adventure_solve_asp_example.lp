turn_limit(15).

turn(0).

{ turn(TURN+1) } 1 :- turn(TURN), not turn_limit(TURN).

at_t(0,kitchen1floor,kitchen1).
at_t(0,pantry1floor,pantry1).

at_t(0,hallway1floor,hallway1).
at_t(0,livingroom1floor,livingroom1).
at_t(0,broomcloset1floor,broomcloset1).
at_t(0,table1,livingroom1).

at_t(0,counter1,kitchen1).
at_t(0,refrigerator1,pantry1).
at_t(0,shelf1,kitchen1).
at_t(0,freezer1,pantry1).

at_t(0,pottedplant1,hallway1).
at_t(0,chair1,livingroom1).
at_t(0,couch1,livingroom1).
at_t(0,broom1,broomcloset1).

at_t(0,sandwich1,pantry1).
at_t(0,apple1,pantry1).
at_t(0,banana1,pantry1).

at_t(0,player1,broomcloset1).

room(kitchen1,kitchen).
room(pantry1,pantry).

room(hallway1,hallway).
room(livingroom1,livingroom).
room(broomcloset1,broomcloset).

exit(kitchen1,pantry1).
exit(pantry1,kitchen1).

exit(kitchen1,hallway1).
exit(hallway1,kitchen1).
exit(hallway1,livingroom1).
exit(hallway1,broomcloset1).
exit(livingroom1,hallway1).
exit(broomcloset1,hallway1).

type(player1,player).
type(kitchen1floor,floor).
type(pantry1floor,floor).

type(hallway1floor,floor).
type(livingroom1floor,floor).
type(broomcloset1floor,floor).
type(table1,table).

type(counter1,counter).
type(refrigerator1,refrigerator).
type(shelf1,shelf).
type(freezer1,freezer).

type(pottedplant1,pottedplant).
type(chair1,chair).
type(couch1,couch).
type(broom1,broom).

type(sandwich1,sandwich).
type(apple1,apple).
type(banana1,banana).
support(kitchen1floor).
support(pantry1floor).

support(hallway1floor).
support(livingroom1floor).
support(broomcloset1floor).
support(table1).

support(counter1).
support(shelf1).

on_t(0,broom1,broomcloset1floor).
on_t(0,pottedplant1,hallway1floor).

container(refrigerator1).
container(freezer1).
in_t(0,banana1,refrigerator1).
in_t(0,apple1,refrigerator1).
in_t(0,sandwich1,refrigerator1).
openable(refrigerator1).
closed_t(0,refrigerator1).
openable(freezer1).
closed_t(0,freezer1).

takeable(pottedplant1).
takeable(broom1).

takeable(sandwich1).
takeable(apple1).
takeable(banana1).

movable(pottedplant1).
movable(broom1).

movable(sandwich1).
movable(apple1).
movable(banana1).

needs_support(pottedplant1).
needs_support(broom1).

needs_support(sandwich1).
needs_support(apple1).
needs_support(banana1).


% go action
{ action_t(TURN,go,TARGET):exit(ROOM,TARGET) } 1 :- turn(TURN), at_t(TURN,player1,ROOM), not turn_limit(TURN).
at_t(TURN+1,player1,TARGET) :- action_t(TURN,go,TARGET).
at_t(TURN+1,player1,ROOM) :- turn(TURN), at_t(TURN,player1,ROOM), not action_t(TURN,go,_).
% things in inventory move with player:
at_t(TURN+1,THING,TARGET) :- action_t(TURN,go,TARGET), in_t(TURN,THING,inventory).
% things not in inventory stay in the room they are:
at_t(TURN+1,THING,ROOM) :- turn(TURN), at_t(TURN,THING,ROOM), not in_t(TURN,THING,inventory), not turn_limit(TURN), THING != player1.

% open action
{ action_t(TURN,open,THING):at_t(TURN,THING,ROOM),closed_t(TURN,THING) } 1 :- turn(TURN), at_t(TURN,player1,ROOM), not turn_limit(TURN).
open_t(TURN+1,THING) :- action_t(TURN,open,THING).
open_t(TURN+1,THING) :- turn(TURN), open_t(TURN,THING), not action_t(TURN,close,THING).

% close action
{ action_t(TURN,close,THING):at_t(TURN,THING,ROOM),open_t(TURN,THING) } 1 :- turn(TURN), at_t(TURN,player1,ROOM), not turn_limit(TURN).
closed_t(TURN+1,THING) :- action_t(TURN,close,THING).
closed_t(TURN+1,THING) :- turn(TURN), closed_t(TURN,THING), not action_t(TURN,open,THING).

% take action
{
action_t(TURN,take,THING):at_t(TURN,THING,ROOM),takeable(THING),in_t(TURN,THING,CONTAINER),open_t(TURN,CONTAINER),at_t(TURN,player1,ROOM);
action_t(TURN,take,THING):at_t(TURN,THING,ROOM),takeable(THING),on_t(TURN,THING,SUPPORT),support(SUPPORT),at_t(TURN,player1,ROOM)
 } 1 :- turn(TURN), at_t(TURN,player1,ROOM), not turn_limit(TURN).
% things taken are in inventory in next turn:
in_t(TURN+1,THING,inventory) :- action_t(TURN,take,THING).
% things remain in container unless taken, ignore inventory (which is not a 'container', but acts similarly and moves with the player):
in_t(TURN+1,THING,TARGET) :- turn(TURN), in_t(TURN,THING,TARGET), not action_t(TURN,take,THING), TARGET != inventory.
% things remain on supports unless taken:
on_t(TURN+1,THING,TARGET) :- turn(TURN), on_t(TURN,THING,TARGET), not action_t(TURN,take,THING).

% put action
{
% put thing from open container in open container:
action_t(TURN,put,THING,TARGET):at_t(TURN,THING,ROOM),at_t(TURN,player1,ROOM),at_t(TURN,TARGET,ROOM),at_t(TURN,SOURCE,ROOM),movable(THING),container(SOURCE),in_t(TURN,THING,SOURCE),open_t(TURN,SOURCE),container(TARGET),open_t(TURN,TARGET);
% put thing from open container on support:
action_t(TURN,put,THING,TARGET):at_t(TURN,THING,ROOM),at_t(TURN,player1,ROOM),at_t(TURN,TARGET,ROOM),at_t(TURN,SOURCE,ROOM),movable(THING),container(SOURCE),in_t(TURN,THING,SOURCE),open_t(TURN,SOURCE),support(TARGET);
% put thing from support on support:
action_t(TURN,put,THING,TARGET):at_t(TURN,THING,ROOM),at_t(TURN,player1,ROOM),at_t(TURN,TARGET,ROOM),support(TARGET),movable(THING),on_t(TURN,THING,SOURCE),support(SOURCE);
% put thing from inventory in open container:
action_t(TURN,put,THING,TARGET):at_t(TURN,THING,ROOM),at_t(TURN,player1,ROOM),at_t(TURN,TARGET,ROOM),movable(THING),in_t(TURN,THING,inventory),container(TARGET),open_t(TURN,TARGET);
% put thing from inventory on support:
action_t(TURN,put,THING,TARGET):at_t(TURN,THING,ROOM),at_t(TURN,player1,ROOM),at_t(TURN,TARGET,ROOM),movable(THING),in_t(TURN,THING,inventory),support(TARGET)
 } 1 :- turn(TURN), at_t(TURN,player1,ROOM), not turn_limit(TURN).
% things put are in container target or on support target in next turn:
in_t(TURN+1,THING,TARGET) :- turn(TURN), action_t(TURN,put,THING,TARGET), container(TARGET).
on_t(TURN+1,THING,TARGET) :- turn(TURN), action_t(TURN,put,THING,TARGET), support(TARGET).
% things remain in inventory next turn unless put somewhere this turn:
in_t(TURN+1,THING,inventory) :- turn(TURN), in_t(TURN,THING,inventory), not action_t(TURN,put,THING,_).


% only one action per turn:
:- { action_t(TURN,_,_);action_t(TURN,_,_,_) } > 1, turn(TURN).

% every turn must have an action:
:- { action_t(TURN,_,_);action_t(TURN,_,_,_) } = 0, turn(TURN).

% goal:
:- not on_t(_,sandwich1,table1).

% optimize for minimum number of turns:
#minimize{1,X:turn(X)}.

% only return certain atoms:
#show turn/1.
#show action_t/3.
#show action_t/4.
#show open_t/2.
#show closed_t/2.
#show at_t/3.
#show in_t/3.
#show on_t/3.