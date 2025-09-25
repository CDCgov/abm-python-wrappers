from cfa.cloudops import CloudClient

client = CloudClient(dotenv_path = ".env")
print(client)