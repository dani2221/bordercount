using System;
using System.ComponentModel.DataAnnotations;

namespace borderCounterApi.DTOs.BorderInformationInterval
{
    public class BorderInformationIntervalRequest
    {
        [Required(AllowEmptyStrings = false)]
        public string? Border { get; set; }

        [Required(AllowEmptyStrings = false)]
        public List<int>? SpeedLanes { get; set; }

        [Required(AllowEmptyStrings = false)]
        public List<int>? CarLanes { get; set; }

        [Required()]
        public int PrevMinutes { get; set; }
    }
}

