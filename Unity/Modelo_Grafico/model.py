from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import json

# Importamos las clases que se requieren para manejar los agentes (Agent) y su entorno (Model).
# Cada modelo puede contener múltiples agentes.
from mesa import Agent, Model

from mesa.batchrunner import batch_run

from mesa.space import PropertyLayer

# Debido a que necesitamos que existe un solo agente por celda, elegimos ''SingleGrid''.
from mesa.space import SingleGrid
from mesa.space import MultiGrid

# Con ''RandomActivation'', hacemos que todos los agentes se activen ''al mismo tiempo''.
from mesa.time import RandomActivation

# Con ''RandomActivation'', hacemos que todos los agentes se activen ''al mismo tiempo''.
from mesa.time import BaseScheduler

# Haremos uso de ''DataCollector'' para obtener información de cada paso de la simulación.
from mesa.datacollection import DataCollector

# matplotlib lo usaremos crear una animación de cada uno de los pasos
# del modelo.
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**128

# seaborn lo usaremos desplegar una gráficas más ''vistosas'' de
# nuestro modelo
import seaborn as sns

import random


# Importamos los siguientes paquetes para el mejor manejo de valores numéricos.
import numpy as np
import pandas as pd

# Definimos otros paquetes que vamos a usar para medir el tiempo de ejecución de nuestro algoritmo.
import time
import datetime

import heapq

#Clase de los dinosaurios
class Dino(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.id = unique_id
        self.position = self.pos
        self.isCarring = False
        self.advancePoints = 4
        self.addedAdvancePoints = 0
        self.movements = []
        self.routes = []



    def step(self):
      #print("agente: {} pos: {}".format(self.id, self.pos))
      self.move()

    def move(self):
      if self.isCarring:

        ap = np.random.randint(1, (self.advancePoints + self.addedAdvancePoints)+1)
        routes = self.get_exit(self.model.entrance)
        del routes[0]
        
        #print(self.pos)

        self.actionMove(1, routes)
        self.isFire()

        if len(routes) == 0:
          self.isCarring = False
          self.model.rescatadas += 1

      else:
        if not(len(self.model.poi) == 0):
          ap = np.random.randint(1, (self.advancePoints + self.addedAdvancePoints)+1)
          routes = self.get_directions_poi(self.model.poi)
          del routes[0]
          #print(routes)
          #print(self.pos)

          self.actionMove(1, routes)
          self.isPOI()
          self.isFire()
          self.position = self.pos

    def get_exit(self, array):
      #print("Get exit")

      temp = []
      for i in range(len(array)):
        temp.append(tuple([array[i][0]-1, array[i][1]-1]))
      #print(temp)
      return self.dijkstra(self.model.grafo, self.pos, temp)

    def get_directions_poi(self, array):
      temp = []
      for i in range(len(array)):
        temp.append(tuple([array[i][0][0]-1, array[i][0][1]-1]))

      return self.dijkstra(self.model.grafo, self.pos, temp)

    def actionMoveVictim(self):
      self.isCarring = True
      self.model.POI.remove([tuple([self.pos[0]+1, self.pos[0]+1]), True])


      pass

    def actionMove(self, ap, ruta):
      if not(len(ruta) == 0):
        flag = self.isWall(self.pos, ruta[0])
        #print(flag)
    
    def isFire(self):
      posicion = tuple([self.pos[0]+1, self.pos[1]+1])
      if posicion in self.model.fire:
        #print("Casilla en llamas")
        #print(self.model.fire)
        del self.model.fire[self.model.fire.index(posicion)]
      elif posicion in self.model.smokes:
        del self.model.smokes[self.model.smokes.index(posicion)]

    def isPOI(self):
      position = tuple([self.pos[0]+1, self.pos[1]+1])
      #print(position)
      #print(self.model.poi)
      for poi in self.model.poi:
        if position == poi[0]:
          if poi[1]:
            self.isCarring = True
            self.model.poi.remove([position, True])
            #print("Es una victima")
          else:
            #print("Es una falsa alarma")
            self.model.poi.remove([position, False])

    def isWall(self, cell, next):
      #print(self.model.map)
      direction = self.model.map[cell[0]][cell[1]]

      result = ((cell[0]-next[0]), (cell[1]-next[1]))


      #Movimiento abajo hacia arriba
      if result == (1, 0):
        if direction[0] == "1": # Si hay un muro en la direcion a moverse
          if self.model.isDoor(self.pos, next, 0): #Si hay una puerta en la celda
            #Modifica los valores de las paredes en el mapa, abriendo la puerta
            self.model.map[cell[0]][cell[1]] = "0" + direction[1:]

            direction = self.model.map[next[0]][next[1]]
            self.model.map[next[0]][next[1]] = direction[:-2] + "0" + direction[-1]
            #print("Intento abrir una puerta en la posicion abajo hacia arriba")
            #print(self.model.map[4])
            #Se mueve
            self.model.grid.move_agent(self, next)
            return True
          else:
            #print("abajo hacia arriba {}".format(direction))
            self.model.damageWall(2)
            self.model.destroyWalls(tuple([next[0]+1, next[1]+1]), tuple([cell[0]+1, cell[1]+1]))
            self.model.destroyWalls(tuple([next[0]+1, next[1]+1]), tuple([cell[0]+1, cell[1]+1]))
            return False

        else:
          self.model.grid.move_agent(self, next)

          return True

      #Movimiento derecha a izquierda
      elif result == (0, 1):
        if direction[1] == "1":
          if self.model.isDoor(self.pos, next, 3): #Si hay una puerta en la celda
            #Modifica los valores de las paredes en el mapa, abriendo la puerta
            self.model.map[cell[0]][cell[1]] = direction[0] + "0" + direction[2:]

            direction = direction = self.model.map[next[0]][next[1]]
            self.model.map[next[0]][next[1]] = direction[:-1] + "0"

            #Se mueve
            self.model.grid.move_agent(self, next)
            return True
          else:
            #print("derecha a izquierda {}".format(direction))
            self.model.damageWall(2)
            self.model.destroyWalls(tuple([next[0]+1, next[1]+1]), tuple([cell[0]+1, cell[1]+1]))
            self.model.destroyWalls(tuple([next[0]+1, next[1]+1]), tuple([cell[0]+1, cell[1]+1]))
            return False

        else:
          self.model.grid.move_agent(self, next)

          return True

      #Movimiento arriba hacia abajo
      elif result == (-1, 0): #Horizontal
        if direction[2] == "1":
          if self.model.isDoor(self.pos, next, 1): #Si hay una puerta en la celda
            #Modifica los valores de las paredes en el mapa, abriendo la puerta
            self.model.map[cell[0]][cell[1]] = direction[:-2] + "0" + direction[-1]

            direction = self.model.map[next[0]][next[1]]
            self.model.map[next[0]][next[1]] =  "0" + direction[1:]


            #Se mueve
            self.model.grid.move_agent(self, next)
            return True
          else:
            #print("arriba hacia abajo {}".format(direction))
            self.model.damageWall(2)
            self.model.destroyWalls(tuple([cell[0]+1, cell[1]+1]), tuple([next[0]+1, next[1]+1]))
            self.model.destroyWalls(tuple([cell[0]+1, cell[1]+1]), tuple([next[0]+1, next[1]+1]))
            return False

        else:
          self.model.grid.move_agent(self, next)

          return True

      #Movimiento izq a derecha
      elif result == (0, -1):
        if direction[3] == "1":#Vertical
          if self.model.isDoor(self.pos, next, 2): #Si hay una puerta en la celda
            #Modifica los valores de las paredes en el mapa, abriendo la puerta
            self.model.map[cell[0]][cell[1]] = direction[:-1] + "0"

            direction = self.model.map[next[0]][next[1]]
            self.model.map[next[0]][next[1]] = direction[0] + "0" + direction[2:]

            #Se mueve
            self.model.grid.move_agent(self, next)
            return True
          else:
            #print("izquierda hacia derecha {}".format(direction))
            self.model.damageWall(2)
            self.model.destroyWalls(tuple([cell[0]+1, cell[1]+1]), tuple([next[0]+1, next[1]+1]))
            self.model.destroyWalls(tuple([cell[0]+1, cell[1]+1]), tuple([next[0]+1, next[1]+1]))
            return False

        else:
          self.model.grid.move_agent(self, next)

          return True
      else:
        return False

    # Función para encontrar el camino más corto usando el algoritmo de Dijkstra
    def dijkstra(self, grafo, inicio, puntos_interes):
      # Usamos una cola de prioridad para manejar los nodos a explorar
      cola = [(0, inicio)]  # (distancia acumulada, nodo actual)
      distancias = {nodo: float('inf') for nodo in grafo}  # Inicializamos las distancias con infinito
      distancias[inicio] = 0
      previos = {nodo: None for nodo in grafo}  # Para reconstruir el camino
      visitados = set()

      while cola:
          distancia_actual, nodo_actual = heapq.heappop(cola)
          if nodo_actual in visitados:
              continue
          visitados.add(nodo_actual)

          # Si llegamos a un punto de interés, terminamos
          if nodo_actual in puntos_interes:
              break

          for vecino, peso in grafo[nodo_actual]:
              nueva_distancia = distancia_actual + peso
              if nueva_distancia < distancias[vecino]:
                  distancias[vecino] = nueva_distancia
                  previos[vecino] = nodo_actual
                  heapq.heappush(cola, (nueva_distancia, vecino))


      # Buscar el punto de interés más cercano
      punto_cercano = min(puntos_interes, key=lambda p: distancias[p])
      # Reconstruir el camino desde el punto más cercano
      camino = []
      nodo = punto_cercano
      while nodo is not None:
          camino.append(nodo)
          nodo = previos[nodo]

      return camino[::-1]  # Devolver el camino en orden correcto

def get_grid(model):
  temp = []
  for i in model.poi:
    temp.append(i[0])

  grid = np.zeros( (model.grid.width, model.grid.height) )
  for (content, (x, y)) in model.grid.coord_iter():
      if (content):
        grid[x][y] = 1
      elif (x+1, y+1) in temp:
        grid[x][y] = 2
      
      elif tuple([x+1, y+1]) in model.smokes:
        grid[x][y] = 3
      elif (x+1, y+1) in model.fire:
        grid[x][y] = 4
      else:
        grid[x][y] = 0

  return grid

#Modelo del tablero
class Board(Model):
  def __init__(self, width, height, array):
    self.grid = MultiGrid(width, height, True)
    self.schedule = BaseScheduler(self)
    self.running = True
    
    self.rescatadas = 0 #Victimas rescatadas
    self.muertos = 0 #Victimas muertas
    self.DP = 0 #Damage points de los muros
    self.victimas = 8 
    self.falsaAlarma = 4 

    self.JsonVictimas = [] #Coordenadas de los POI con Victimas
    self.JsonFalsaAlarma = [] #Coordenadas con los POI 
    self.map = array[0] #Mapa de las paredes del tablero
    self.poi = array[1] #Puntos de interes
    self.fire = array[2] #Casillas con fuego
    self.doors = array[3] #Casillas con puertas
    self.entrance = array[4] #Casillas con entradas
    self.smokes = [] #Casillas con humo
    self.grafo = {} #Grafo de las casillas
    self.make_graph()

    self.DW = {} #Damage walls de los muros

    
    self.datacollector = DataCollector(model_reporters = {"Map": "map", "Victimas": "JsonVictimas", "Fire": "fire", "Smokes": "smokes", "FA":"JsonFalsaAlarma", "Muertos":"muertos", "Rescatados":"rescatadas","DP":"DP"}, agent_reporters = {"Pos": "pos"})
    """
    self.datacollector = DataCollector(
      model_reporters = {"Grid": get_grid}
    )
    """
    self.start = [tuple([0,5]), tuple([2,0]), tuple([3,7]), tuple([5,2])]

    #Crea los agentes
    for i in range(0, 6, 1):
      agent = Dino(i, self)
      self.grid.place_agent(agent, self.start[np.random.randint(0,4)])
      #self.grid.place_agent(agent, (5, 7))
      self.schedule.add(agent)

  def step(self):
    print("\nStep: " + str(self.schedule.steps))
    #print("Walls: "+str(self.map))
    #print("Smokes: "+str(self.smokes))
    #print("Fire: " + str(self.fire))
    #print("POI: " + str(self.poi))
    print("Rescatadas: " + str(self.rescatadas))
    print("Muertos: "+str(self.muertos))

    #print("Victimas: "+ str(self.victimas))
    #print("Falsa Alarma: " + str(self.falsaAlarma))
    print("DP: " + str(self.DP))
    print("DW: "+str(self.DW))
    
    """
    print("Door: "+ str(self.doors))
   
    """
    self.traductor()
    """
    if self.muertos >= 4:
      print("----GAME OVER----El juego ha terminado porque han muerto 4 Victimas")
      self.running = False
    elif self.DP >= 24:
      print("----GAME OVER----El juego ha terminado porque el edificio se ha derrumbado")
      self.running = False
    elif self.rescatadas >= 7:
      print("----SE GANO EL JUEGO---- Se han salvado 7 victimas exitosamente")
      self.running = False
    """
    self.datacollector.collect(self)
    self.schedule.step()
    self.setFire()
    self.smoke2Fire()
    self.reveal()
    self.knockOut()
    
    if len(self.poi) < 3:
      self.refillPOI()
    
    

    #print("Smoke: " + str(self.smokes))
  def traductor(self):
    temp1 = [] #Victima
    temp2 = [] #Falsa alarma
    for i in self.poi:
      if i[1]:
        temp1.append(i[0])
      else:
        temp2.append(i[0])

    self.JsonVictimas = temp1
    self.JsonFalsaAlarma = temp2

  def smoke2Fire(self):
    for smoke in self.smokes:
      smokeNeighbors = self.grid.get_neighborhood(tuple([smoke[0]-1, smoke[1]-1]), moore=False, include_center=False)
      for i in smokeNeighbors:
        if tuple([i[0]+1, i[1]+1]) in self.fire:
          self.smokes.remove(smoke)
          self.fire.append(smoke)
          break

  #Revela el POI cuando un fuego se coloca en su posicion
  def reveal(self):
    for element in self.poi:
      if element[0] in self.fire:
        if element[1]:
          print("La revelacion era una victima!!!")
          self.muertos += 1
          self.poi.remove(element)
        else:
          print("La revelacion era una falsa alarma")

  def refillPOI(self):
    temp = []
    for i in self.poi:
      temp.append(i[0])

    # Si aun hay ambos
    if self.victimas > 0 and self.falsaAlarma > 0:
      coin = np.random.randint(0, 2)
      if coin == 0:
        while True:
          new = tuple([np.random.randint(1, 7), np.random.randint(1, 9)])
          if new not in temp:
            self.poi.append([new, False])
            self.falsaAlarma -= 1

            if new in self.smokes:
              self.smokes.remove(new)
            elif new in self.fire:
              self.fire.remove(new)

            break
      else:
        while True:
          new = tuple([np.random.randint(1, 7), np.random.randint(1, 9)])
          if new not in temp:
            self.poi.append([new, True])
            self.victimas -= 1

            if new in self.smokes:
              self.smokes.remove(new)
            elif new in self.fire:
              self.fire.remove(new)

            break

    # Si no hay victimas pero si falsas alarmas
    elif self.victimas == 0 and self.falsaAlarma > 0:
      while True:
        new = tuple([np.random.randint(1, 7), np.random.randint(1, 9)])
        if new not in temp:
          self.poi.append([new, False])
          self.falsaAlarma -= 1

          if new in self.smokes:
            self.smokes.remove(new)
          elif new in self.fire:
            self.fire.remove(new)

          break

    # Si hay victimas pero no falsas alarmas
    elif self.victimas > 0 and self.falsaAlarma == 0:
      while True:
        new = tuple([np.random.randint(1, 7), np.random.randint(1, 9)])
        if new not in temp:
          self.poi.append([new, True])
          self.victimas -= 1

          if new in self.smokes:
            self.smokes.remove(new)
          elif new in self.fire:
            self.fire.remove(new)

          break

  def make_graph(self):

    puertas = []
    for i in range(len(self.doors)):
      temp = []
      for j in range(2):
        temp.append(tuple([self.doors[i][j][0]-1, self.doors[i][j][1]-1]))
      puertas.append(temp)
    #print(puertas)

    self.grafo = self.crear_grafo(self.map, puertas)
    """
    # Imprimir el grafo resultante para revisar
    for nodo, conexiones in self.grafo.items():
        print(f"Nodo {nodo}: {conexiones}")
    """
  # Función para crear el grafo
  def crear_grafo(self, mapa, puertas):
    filas = len(mapa)
    columnas = len(mapa[0])
    grafo = {}

    # Crear grafo básico con pesos según paredes
    for i in range(filas):
        for j in range(columnas):
            nodo = (i, j)
            grafo[nodo] = []

            paredes = mapa[i][j]

            # Arriba
            if i > 0:
                peso = 4 if paredes[0] == '1' or mapa[i - 1][j][2] == '1' else 1
                grafo[nodo].append(((i - 1, j), peso))

            # Izquierda
            if j > 0:
                peso = 4 if paredes[1] == '1' or mapa[i][j - 1][3] == '1' else 1
                grafo[nodo].append(((i, j - 1), peso))

            # Abajo
            if i < filas - 1:
                peso = 4 if paredes[2] == '1' or mapa[i + 1][j][0] == '1' else 1
                grafo[nodo].append(((i + 1, j), peso))

            # Derecha
            if j < columnas - 1:
                peso = 4 if paredes[3] == '1' or mapa[i][j + 1][1] == '1' else 1
                grafo[nodo].append(((i, j + 1), peso))

    # Ajustar las conexiones de las puertas para tener peso 2
    for puerta in puertas:
        nodo1, nodo2 = puerta
        # Remover conexiones existentes entre los nodos de la puerta
        grafo[nodo1] = [(nodo, peso) for nodo, peso in grafo[nodo1] if nodo != nodo2]
        grafo[nodo2] = [(nodo, peso) for nodo, peso in grafo[nodo2] if nodo != nodo1]

        # Agregar las conexiones con peso 2
        grafo[nodo1].append((nodo2, 2))
        grafo[nodo2].append((nodo1, 2))
    
    return grafo

  def destroyWalls(self, cell, next):
    #print("cell: {}, next: {}".format(cell, next))
    result = tuple([cell[0]-next[0], cell[1]-next[1]])
    if result == (1, 0) or result == (-1, 0):
      #print("is {} in {}".format(tuple([cell, next]), tuple([cell, next]) in self.DW))
      if tuple([cell, next]) in self.DW:
        #print(self.map)
        self.DW[tuple([cell, next])] = 2

        self.map[cell[0]-1][cell[1]-1] = self.map[cell[0]-1][cell[1]-1][:-2] + "0" + self.map[cell[0]-1][cell[1]-1][-1]

        if not(self.grid.out_of_bounds(tuple([next[0]-1, next[1]-1]))):
          self.map[next[0]-1][next[1]-1] = "0" + self.map[next[0]-1][next[1]-1][1:]
        #print(self.map)
        #print("Se rompio una pared {} {}".format(cell, next))
      else:
        self.DW[tuple([cell, next])] = 1
    else:
      #print("is {} in {}".format(tuple([cell, next]), tuple([cell, next]) in self.DW))
      if tuple([cell, next]) in self.DW:
        self.DW[tuple([cell, next])] = 2
        #print(self.map)
        #print(tuple([cell[0]-1, cell[1]-1]))
        self.map[cell[0]-1][cell[1]-1] = self.map[cell[0]-1][cell[1]-1][:-1] + "0"
        if not(self.grid.out_of_bounds(tuple([next[0]-1, next[1]]))):
          #print("next kk: {}".format(tuple([next[0]-1, next[1]])))
          self.map[next[0]-1][next[1]] = self.map[next[0]-1][next[1]][0] + "0" + self.map[next[0]-1][next[1]][2:]
        #print(self.map)
        #print("Se rompio una pared {} {}".format(cell, next))
      else:
        self.DW[tuple([cell, next])] = 1
    #print("Contador de daños: {}".format(self.DW))

  def knockOut(self):
      for agent in self.schedule.agents:
        if tuple([agent.pos[0]+1,agent.pos[1]+1]) in self.fire:
          print("Se noqueo al agente {} en la posicion {} regresando a una entrada".format(agent.id, tuple([agent.pos[0], agent.pos[1]])))
          if agent.isCarring:
            self.muertos += 1
            agent.isCarring = False
            print("Se murio la victima que estaba llevando el agente")
          self.grid.move_agent(agent, self.start[np.random.randint(0,4)])

  def diceRun(self):
    dice1 = np.random.randint(1,7)
    dice2 = np.random.randint(1,7)
    return tuple([dice1, dice2])

  def damageWall(self, amount):
    self.DP = self.DP + amount
    """
    if amount == 1:
      print("Damage wall")
    else:
      print("Damage door")
    """


  def setFire(self):
    newFire = self.diceRun()
    print("Set fire")
    #newFire = (6, 1)
    if newFire in self.smokes:
      print("se reemplazó")
      self.fire.append(newFire)
      self.smokes.remove(newFire)

    elif newFire in self.fire:
      print("explosion in {}".format(newFire))
      fireNeighbors = self.grid.get_neighborhood((newFire[0]-1, newFire[1]-1), moore=False, include_center=False)
      #print(fireNeighbors)

      for i in range(len(fireNeighbors)):
        temp = tuple([newFire[0]-1 - fireNeighbors[i][0], newFire[1]-1 - fireNeighbors[i][1]])
        #print("fuego: {}, next: {}, resta: {}".format((newFire[0]-1, newFire[1]-1), fireNeighbors[i], temp))

        if temp == (0, -1): # ---->
          #print("ex derecha")
          kk = newFire
          while True:
            #print("aqui")
            #print(kk)
            #print("out {}".format(self.grid.out_of_bounds(tuple([kk[0]-1, kk[1]-1]))))
            if self.grid.out_of_bounds(tuple([kk[0]-1, kk[1]-1])):
              break
            if self.map[kk[0]-1][kk[1]-1][3] == "1": #Si se encuentra una pared
              #print(self.isDoor((kk[0]-1, kk[1]-1), (kk[0]-1, kk[1]), 2))
              if self.isDoor((kk[0]-1, kk[1]-1), (kk[0]-1, kk[1]), 2): #Si la casilla es una puerta cerrada
                #print("Entro")
                direction = self.map[kk[0]-1][kk[1]-1]
                self.map[kk[0]-1][kk[1]-1] = direction[:-1] + "0"

                direction = self.map[kk[0]-1][kk[1]]
                self.map[kk[0]-1][kk[1]] = direction[0] + "0" + direction[2:]

                self.doors.remove([tuple([kk[0], kk[1]]), tuple([kk[0], kk[1]+1])])
                self.damageWall(2)
              else:

                self.destroyWalls(tuple([kk[0], kk[1]]), tuple([kk[0], kk[1]+1]))
                self.damageWall(1)


              if kk in self.smokes:
                self.smokes.remove(kk)
                self.fire.append(kk)
              elif kk not in self.fire:
                self.fire.append(kk)

              #print("Salio")
              break
            else:
              #print("Free")
              if self.isDoor((kk[0]-1, kk[1]-1), (kk[0]-1, kk[1]), 2): #Si la casilla es una puerta abierta

                direction = self.map[kk[0]-1][kk[1]-1]
                self.map[kk[0]-1][kk[1]-1] = direction[:-1] + "0"

                direction = self.map[kk[0]-1][kk[1]]
                self.map[kk[0]-1][kk[1]] = direction[0] + "0" + direction[2:]
                self.doors.remove([tuple([kk[0], kk[1]]), tuple([kk[0], kk[1]+1])])

                self.damageWall(2)

                if kk in self.smokes:
                  self.smokes.remove(kk)
                  self.fire.append(kk)
                elif kk not in self.fire:
                  self.fire.append(kk)

              else:

                if kk in self.smokes:
                  self.smokes.remove(kk)
                  self.fire.append(kk)
                  break
                elif kk not in self.fire:
                  self.fire.append(kk)
                  break


            kk = (kk[0], kk[1]+1)

        elif temp == (0, 1):# <----
          #print("ex izq")
          kk = newFire
          while True:
            #print(kk)
            if self.grid.out_of_bounds(tuple([kk[0]-1, kk[1]-1])):
              break
            if self.map[kk[0]-1][kk[1]-1][1] == "1": #Si se encuentra una pared

              if self.isDoor((kk[0]-1, kk[1]-1), (kk[0]-1, kk[1]-2), 3): #Si la casilla es una puerta cerrada
                direction = self.map[kk[0]-1][kk[1]-1]
                self.map[kk[0]-1][kk[1]-1] = direction[0] + "0" + direction[2:]

                direction = self.map[kk[0]-1][kk[1]-2]
                self.map[kk[0]-1][kk[1]-2] = direction[:-1] + "0"
                self.doors.remove([tuple([kk[0], kk[1]-1]), tuple([kk[0], kk[1]])])
                self.damageWall(2)
              else:
                self.destroyWalls(tuple([kk[0], kk[1]-1]), tuple([kk[0], kk[1]]))
                self.damageWall(1)

              if kk in self.smokes:
                self.smokes.remove(kk)
                self.fire.append(kk)
              elif kk not in self.fire:
                self.fire.append(kk)

              break

            else:
              if self.isDoor((kk[0]-1, kk[1]-1), (kk[0]-1, kk[1]-2), 3): #Si la casilla es una puerta cerrada
                direction = self.map[kk[0]-1][kk[1]-1]
                self.map[kk[0]-1][kk[1]-1] = direction[0] + "0" + direction[2:]

                direction = self.map[kk[0]-1][kk[1]-2]
                self.map[kk[0]-1][kk[1]-2] = direction[:-1] + "0"
                self.doors.remove([tuple([kk[0], kk[1]-1]), tuple([kk[0], kk[1]])])

                if kk in self.smokes:
                  self.smokes.remove(kk)
                  self.fire.append(kk)
                elif kk not in self.fire:
                  self.fire.append(kk)

                self.damageWall(2)
              else:
                if kk in self.smokes:
                  self.smokes.remove(kk)
                  self.fire.append(kk)
                  break
                elif kk not in self.fire:
                  self.fire.append(kk)
                  break

            kk = (kk[0], kk[1]-1)

        elif temp == (1, 0):# UP
          #print("ex up")
          kk = newFire
          while True:
            #print(kk)
            
            if self.grid.out_of_bounds(tuple([kk[0]-1, kk[1]-1])):
              break
            if self.map[kk[0]-1][kk[1]-1][0] == "1": #Si se encuentra una pared

              if self.isDoor((kk[0]-1, kk[1]-1), (kk[0]-2, kk[1]-1), 0): #Si la casilla es una puerta cerrada
                direction = self.map[kk[0]-1][kk[1]-1]

                self.map[kk[0]-1][kk[1]-1] = "0" + direction[1:]

                direction = self.map[kk[0]-2][kk[1]-1]
                self.map[kk[0]-2][kk[1]-1] = direction[:-2] + "0" + direction[-1]
                self.doors.remove([tuple([kk[0]-1, kk[1]]), tuple([kk[0], kk[1]])])
                self.damageWall(2)
              else:
                self.destroyWalls(tuple([kk[0]-1, kk[1]]), tuple([kk[0], kk[1]]))
                self.damageWall(1)

              if kk in self.smokes:
                self.smokes.remove(kk)
                self.fire.append(kk)
              elif kk not in self.fire:
                self.fire.append(kk)

              break

            else:

              if self.isDoor((kk[0]-1, kk[1]-1), (kk[0]-2, kk[1]-1), 0): #Si la casilla es una puerta cerrada
                direction = self.map[kk[0]-1][kk[1]-1]
                self.map[kk[0]-1][kk[1]-1] = "0" + direction[1:]

                direction = self.map[kk[0]-2][kk[1]-1]
                self.map[kk[0]-2][kk[1]-1] =  direction[:-2] + "0" + direction[-1]
                self.doors.remove([tuple([kk[0]-1, kk[1]]), tuple([kk[0], kk[1]])])
                self.damageWall(2)

                if kk in self.smokes:
                  self.smokes.remove(kk)
                  self.fire.append(kk)
                elif kk not in self.fire:
                  self.fire.append(kk)

              else:
                if kk in self.smokes:
                  self.smokes.remove(kk)
                  self.fire.append(kk)
                  break
                elif kk not in self.fire:
                  self.fire.append(kk)
                  break

            kk = (kk[0]-1, kk[1])

        elif temp == (-1, 0):# DOWN
          #print("ex down")
         
          kk = newFire
          while True:
            #print(kk)
            
            #print("out {}".format(self.grid.out_of_bounds(tuple([kk[0]-1, kk[1]-1]))))
            if self.grid.out_of_bounds(tuple([kk[0]-1, kk[1]-1])):
              break
            #print(self.map[kk[0]-1][kk[1]-1])
            if self.map[kk[0]-1][kk[1]-1][2] == "1": #Si se encuentra una pared

              if self.isDoor((kk[0]-1, kk[1]-1), (kk[0], kk[1]-1), 1): #Si la casilla es una puerta cerrada

                direction = self.map[kk[0]-1][kk[1]-1]
                self.map[kk[0]-1][kk[1]-1] =  direction[:-2] + "0" + direction[-1]

                direction = self.map[kk[0]][kk[1]-1]
                self.map[kk[0]][kk[1]-1] = "0" + direction[1:]

                self.doors.remove([tuple([kk[0], kk[1]]), tuple([kk[0]+1, kk[1]])])
                self.damageWall(2)
              else:
                self.destroyWalls(tuple([kk[0], kk[1]]), tuple([kk[0]+1, kk[1]]))
                self.damageWall(1)

              if kk in self.smokes:
                self.smokes.remove(kk)
                self.fire.append(kk)
              elif kk not in self.fire:
                self.fire.append(kk)
              break
            else:

              if self.isDoor((kk[0]-1, kk[1]-1), (kk[0], kk[1]-1), 1): #Si la casilla es una puerta abierta
                direction = self.map[kk[0]-1][kk[1]-1]
                self.map[kk[0]-1][kk[1]-1] =  direction[:-2] + "0" + direction[-1]

                direction = self.map[kk[0]][kk[1]-1]
                self.map[kk[0]][kk[1]-1] = "0" + direction[1:]
                self.doors.remove([tuple([kk[0], kk[1]]), tuple([kk[0]+1, kk[1]])])

                if kk in self.smokes:
                  self.smokes.remove(kk)
                  self.fire.append(kk)
                elif kk not in self.fire:
                  self.fire.append(kk)
                self.damageWall(2)

              else:
                if kk in self.smokes:
                  self.smokes.remove(kk)
                  self.fire.append(kk)
                  break
                elif kk not in self.fire:
                  self.fire.append(kk)
                  break

            kk = (kk[0]+1, kk[1])

    else:
      print("New smoke")
      self.smokes.append(newFire)


  def isDoor(self, pos, next, flag):
    if flag == 0:
      for door in self.doors:
        objective = tuple([door[0][0]-1, door[0][1]-1])
        casilla = tuple([door[1][0]-1, door[1][1]-1])
        if pos == casilla and next == objective:
          #print("Abrio la concha abajo hacia arriba")
          return True
    elif flag == 1:
      for door in self.doors:
        casilla = tuple([door[0][0]-1, door[0][1]-1])
        objective = tuple([door[1][0]-1, door[1][1]-1])
        if pos == casilla and next == objective:
          #print("Abrio la concha arriba hacia abajo")
          return True
    elif flag == 2:
      for door in self.doors:
        casilla = tuple([door[0][0]-1, door[0][1]-1])
        objective = tuple([door[1][0]-1, door[1][1]-1])
        #print("pos {} casilla {}   next {} Obj {}".format(pos,casilla,next,objective))
        if pos == casilla and next == objective:
          #print("Abrio la concha izq hacia derecha")
          #print(True)
          return True
    elif flag == 3:
      for door in self.doors:
        objective = tuple([door[0][0]-1, door[0][1]-1])
        casilla = tuple([door[1][0]-1, door[1][1]-1])
        if pos == casilla and next == objective:
          #print("Abrio la concha derecha hacia izq")
          return True
    #print(False)
    return False



WIDTH = 6
HEIGHT = 8
miArchivo = open("configInicial.txt","r")

mapa = []

for i in range(6):
    linea = miArchivo.readline()
    row = []
    for i in range(8):
        row.append("".join(list(linea)[:4]))
        linea = linea[5:]
    mapa.append(row)


#print(mapa)

poi = []
for i in range(3):
    temp = []
    linea = miArchivo.readline()
    list(linea)
    temp.append(tuple([int(linea[0]), int(linea[2])]))
    if linea[4] == "v":
        temp.append(True)
    else:
        temp.append(False)
    poi.append(temp)

#print(poi)

fire = []
for i in range(10):
    linea = miArchivo.readline()
    list(linea)
    fire.append(tuple([int(linea[0]), int(linea[2])]))

#print(fire)

doors = []
for i in range(8):
    temp = []
    linea = miArchivo.readline()
    list(linea)
    temp.append(tuple([int(linea[0]), int(linea[2])]))
    temp.append(tuple([int(linea[4]), int(linea[6])]))
    doors.append(temp)

#print(doors)

entrance = []
for i in range(4):
    linea = miArchivo.readline()
    list(linea)
    entrance.append(tuple([int(linea[0]), int(linea[2])]))

#print(entrance)

configInicial = [mapa] + [poi] + [fire] + [doors] + [entrance]
print(configInicial)

miArchivo.close()

model = Board(WIDTH, HEIGHT, configInicial)

"""
model.step()
infoTablero = model.datacollector.get_model_vars_dataframe()
print(infoTablero)


for i in range(STEPS):
  model.step()
model.step()

infoBomberos = model.datacollector.get_agent_vars_dataframe().to_json()
print(infoBomberos)
infoTablero = model.datacollector.get_model_vars_dataframe().to_json()
print(infoTablero)


# Obtenemos la información que almacenó el colector, este nos entregará 
# un DataFrame de pandas que contiene toda la información.
all_grid = model.datacollector.get_model_vars_dataframe()
print(all_grid)
# Graficamos la información usando `matplotlib`
# %%capture

fig, axs = plt.subplots(figsize=(7,7))
axs.set_xticks([])
axs.set_yticks([])
patch = plt.imshow(all_grid.iloc[0,0], cmap=plt.cm.tab20)

def animate(i):
    patch.set_data(all_grid.iloc[i,0])

anim = animation.FuncAnimation(fig, animate, frames=STEPS)

"""
class Server(BaseHTTPRequestHandler):
    
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
    def do_GET(self):
        
        model.step()
        infoBomberos = model.datacollector.get_agent_vars_dataframe().to_json()
        infoTablero = model.datacollector.get_model_vars_dataframe()
        copiaMapa = infoTablero.to_json(orient='records', index=False)
        infoTablero = infoTablero.to_json()
        
        # Cargar el JSON en una lista de diccionarios
        data = json.loads(copiaMapa)
        
        # Extraer solo la columna "Map"
        map_data = data[0]["Map"]
        
        
        # Convertir la lista en un diccionario con "temp" como clave
        diccionario = {"Paredes": map_data}
        
        # Convertir el diccionario a formato JSON
        json_resultado = json.dumps(diccionario, indent=4)
        
        #print(json_resultado)
        
        self._set_response()
        self.wfile.write("{}\n{}\n{}".format(infoBomberos, infoTablero, json_resultado).encode('utf-8'))
        
        
        

    def do_POST(self):
        model.step()
        infoBomberos = model.datacollector.get_agent_vars_dataframe().to_json()
        infoTablero = model.datacollector.get_model_vars_dataframe()
        copiaMapa = infoTablero.to_json(orient='records', index=False)
        infoTablero = infoTablero.to_json()
        
        # Cargar el JSON en una lista de diccionarios
        data = json.loads(copiaMapa)
        
        # Extraer solo la columna "Map"
        map_data = data[0]["Map"]
        
        
        # Convertir la lista en un diccionario con "temp" como clave
        diccionario = {"Paredes": map_data}
        
        # Convertir el diccionario a formato JSON
        json_resultado = json.dumps(diccionario, indent=4)
        
        print(json_resultado)
        
        self._set_response()
        self.wfile.write("{}\n{}\n{}".format(infoBomberos, infoTablero, json_resultado).encode('utf-8'))
        


def run(server_class=HTTPServer, handler_class=Server, port=8585):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info("Starting httpd...\n") # HTTPD is HTTP Daemon!
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:   # CTRL+C stops the server
        pass
    httpd.server_close()
    logging.info("Stopping httpd...\n")

if __name__ == '__main__':
    from sys import argv
    
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
