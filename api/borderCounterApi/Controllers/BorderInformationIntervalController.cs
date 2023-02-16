using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using borderCounterApi.Services;
using Microsoft.AspNetCore.Mvc;
using borderCounterApi.DTOs.BorderInformationInterval;

// For more information on enabling Web API for empty projects, visit https://go.microsoft.com/fwlink/?LinkID=397860

namespace borderCounterApi.Controllers
{
    [Route("api/[controller]")]
    public class BorderInformationIntervalController : Controller
    {
        private readonly IBorderInformationIntervalService borderInformationIntervalService;

        public BorderInformationIntervalController(IBorderInformationIntervalService borderInformationIntervalService)
        {
            this.borderInformationIntervalService = borderInformationIntervalService;
        }

        // GET api/BorderInformationInterval/{id}
        [HttpGet("{id}")]
        public async Task<BorderInformationIntervalResponse> Get(Guid id)
        {
            return await this.borderInformationIntervalService.Get(id);
        }

        // GET api/BorderInformationInterval
        [HttpGet]
        public async Task<List<BorderInformationIntervalResponse>> GetLast([FromQuery] string border, [FromQuery] int limit, [FromQuery] int skipAggregate)
        {
            return await this.borderInformationIntervalService.GetLast(border, limit, skipAggregate);
        }

        // POST api/BorderInformationInterval
        [HttpPost]
        public async Task<Guid> Post([FromBody] BorderInformationIntervalRequest request)
        {
            return await this.borderInformationIntervalService.Add(request);
        }
    }
}

