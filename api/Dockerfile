# https://hub.docker.com/_/microsoft-dotnet
FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /source

# copy csproj and restore as distinct layers
COPY borderCounterApi/*.csproj .
RUN dotnet restore --use-current-runtime  

# copy everything else and build app
COPY borderCounterApi/. .
RUN dotnet publish -c Release -o /app --use-current-runtime --self-contained false

# final stage/image
FROM mcr.microsoft.com/dotnet/aspnet:7.0
WORKDIR /app
COPY --from=build /app .
ENTRYPOINT ["dotnet", "borderCounterApi.dll"]