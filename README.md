## Instruction to create environment

```
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

## Install Dependencies

```
pip install -r requirements.txt
```

## Start RabbitMQ

```
# Run RabbitMQ with management plugin
docker run -d --name rabbitmq \
  -p 5672:5672 \
  -p 15672:15672 \
  rabbitmq:3-management

# Access management UI at: http://localhost:15672
# Default credentials: guest/guest
```







