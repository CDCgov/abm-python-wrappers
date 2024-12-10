#Build Java application
FROM eclipse-temurin:17 as java-build
WORKDIR /app

# Copy the Java JAR file into build storage
COPY ./target/*.jar /app/app.jar

# Second stage: Python with Java runtime
FROM python:3.10.12

# Configure the Python environment
ENV PYTHONFAULTHANDLER=1 \
PYTHONUNBUFFERED=1 \
PYTHONHASHSEED=random \
PIP_NO_CACHE_DIR=off \
PIP_DISABLE_PIP_VERSION_CHECK=on \
PIP_DEFAULT_TIMEOUT=100 \
POETRY_NO_INTERACTION=1 \
POETRY_VIRTUALENVS_CREATE=false \
POETRY_CACHE_DIR='/var/cache/pypoetry' \
POETRY_HOME='/usr/local' \
POETRY_VERSION=1.8.4 \
JAVA_HOME=/opt/java/openjdk

#Add Java to PATH
ENV PATH="${JAVA_HOME}/bin:${PATH}"

#Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://install.python-poetry.org | python3 -

# Verify the installation
RUN poetry --version
COPY --from=eclipse-temurin:17-jre $JAVA_HOME $JAVA_HOME
RUN java -version

# Set up Python application dependencies
WORKDIR /
COPY poetry.lock pyproject.toml ./
RUN poetry install --no-interaction --no-ansi

# Copy the Java application from the first stage
COPY --from=java-build /app/app.jar /app.jar

# Specify the default command
CMD ["java", "-jar", "/app.jar"]
