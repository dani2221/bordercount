using System;
using borderCounterApi.DTOs.BorderInformationInterval;
using borderCounterApi.Helpers;
using borderCounterApi.Models;
using Microsoft.EntityFrameworkCore;

namespace borderCounterApi.Services
{
	public class BorderInformationIntervalService : IBorderInformationIntervalService
	{
        private readonly DataContext dbContext;
		public BorderInformationIntervalService(DataContext dbContext)
		{
            this.dbContext = dbContext;
		}

        public async Task<Guid> Add(BorderInformationIntervalRequest interval)
        {
            var entity = BorderInformationInterval.Build(interval);
            await dbContext.BorderInformationIntervals.AddAsync(entity);
            await dbContext.SaveChangesAsync();
            return entity.Id;
        }

        public async Task<BorderInformationIntervalResponse> Get(Guid guid)
        {
            return BorderInformationIntervalResponse.Build(await dbContext.BorderInformationIntervals.FindAsync(guid));
        }

        public async Task<List<BorderInformationIntervalResponse>> GetLast(string border, int limit, int skipAggregate)
        {
            var results = await dbContext.BorderInformationIntervals
                .Where(x => x.Border == border)
                .OrderByDescending(x => x.Timestamp)
                .Take(limit)
                .Select(x => BorderInformationIntervalResponse.Build(x))
                .ToListAsync();

            if(skipAggregate == 1)
            {
                return results;
            }

            List<BorderInformationIntervalResponse> aggregatedResults = new List<BorderInformationIntervalResponse>();
            for(int i=0; i<results.Count; i += skipAggregate)
            {
                int sumCars = 0;
                int sumPrevMins = 0;
                List<int> carLanes = Enumerable.Repeat(0, results[0].CarLanes.Count).ToList();
                List<int> speedLanes = Enumerable.Repeat(0, results[0].SpeedLanes.Count).ToList();
                foreach (var part in results.Skip(i).Take(skipAggregate))
                {
                    sumCars += part.TotalCars;
                    sumPrevMins += part.PrevMinutes;
                    carLanes = carLanes.Zip(part.CarLanes, (a, b) => a + b).ToList();
                    speedLanes = speedLanes.Zip(part.SpeedLanes, (a, b) => a + b).ToList();
                }
                aggregatedResults.Add(new BorderInformationIntervalResponse()
                {
                    Border = results[0].Border,
                    SpeedLanes = speedLanes,
                    CarLanes = carLanes,
                    Id = Guid.Empty,
                    Timestamp = results[i].Timestamp,
                    PrevMinutes = sumPrevMins,
                    TotalCars = sumCars
                });
            }
            return aggregatedResults;
        }
    }
}

