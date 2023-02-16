using System;
namespace borderCounterApi.DTOs.BorderInformationInterval
{
	public class BorderInformationIntervalResponse
	{
        public Guid Id { get; set; }
        public string Border { get; set; }
        public List<int> SpeedLanes { get; set; }
        public List<int> CarLanes { get; set; }
        public int TotalCars { get; set; }
        public DateTime Timestamp { get; set; }
        public int PrevMinutes { get; set; }

        public static BorderInformationIntervalResponse Build(Models.BorderInformationInterval interval)
        {
            if (interval is null)
            {
                throw new ArgumentNullException(nameof(interval));
            }
            return new BorderInformationIntervalResponse()
            {
                Id = interval.Id,
                SpeedLanes = interval.SpeedLanes,
                Border = interval.Border,
                CarLanes = interval.CarLanes,
                Timestamp = interval.Timestamp,
                TotalCars = interval.TotalCars,
                PrevMinutes = interval.PrevMinutes
            };
        }
    }
}

