using borderCounterApi.DTOs.BorderInformationInterval;
using borderCounterApi.Helpers;

namespace borderCounterApi.Models
{
    public class BorderInformationInterval
	{
		public Guid Id { get; set; }
        public string Border { get; set; }
		public List<int> SpeedLanes { get; set; }
		public List<int> CarLanes { get; set; }
		public int TotalCars { get; set; }
		public DateTime Timestamp { get; set; }
        public int PrevMinutes { get; set; }

        public BorderInformationInterval(string border, List<int> speedLanes, List<int> carLanes, int prevMinutes)
        {
            Border = border;
            SpeedLanes = speedLanes;
            CarLanes = carLanes;
            TotalCars = carLanes.Sum();
            Timestamp = DateTime.UtcNow;
            PrevMinutes = prevMinutes;
        }


        public static BorderInformationInterval Build(BorderInformationIntervalRequest request)
        {
            if (request is null)
            {
                throw new ArgumentNullException(nameof(request));
            }
            return new BorderInformationInterval(request.Border, request.SpeedLanes, request.CarLanes, request.PrevMinutes);
        }
    }
}

