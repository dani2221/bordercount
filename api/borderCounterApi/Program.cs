using AutoMapper;
using borderCounterApi.DTOs.BorderInformationInterval;
using borderCounterApi.Helpers;
using borderCounterApi.Models;
using borderCounterApi.Services;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(
        policy =>
        {
            policy.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader();
        });
});

builder.Services.AddControllers();
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddDbContext<DataContext>();
builder.Services.AddScoped<IBorderInformationIntervalService, BorderInformationIntervalService>();
builder.Services.AddSingleton<IBorderImageService, BorderImageService>();
var config = new MapperConfiguration(cfg => {
    cfg.CreateMap<BorderInformationIntervalRequest, BorderInformationInterval>();
});

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseCors();

app.UseAuthorization();

app.MapControllers();

app.Run();

