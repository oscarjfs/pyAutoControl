# pyAutoControl

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Contributing](../CONTRIBUTING.md)

## About <a name = "about"></a>

Librerias para la implementación de controladores automaticos para procesos usando diferentes algoritmos de control.

## Getting Started <a name = "getting_started"></a>


### Prerequisites

Las librerias que se requieren para su funcionamiento:

```
matplotlib>=3.6.3
numpy>=1.25.1
pandas>=1.5.2
scipy>=1.10.0
```

### Installing

Descargue la carpeta pyAutoControl e incluyala en su proyecto.


## Usage <a name = "usage"></a>

Para incluirla en el proyecto

```
from pyAutoControl.PIDController import PIDController

controller = PIDControler(time_sample=0.1, Kc=1.0)
```